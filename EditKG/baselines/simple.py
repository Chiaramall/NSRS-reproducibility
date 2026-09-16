"""BPR-MF, LightGCN and SGL baselines with full-ranking evaluation."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bpr", "lightgcn", "sgl"], required=True)
    parser.add_argument("--dataset", default="last-fm")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--ssl_weight", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="result")
    parser.add_argument("--validation_only", action="store_true")
    return parser.parse_args()


def read_split(path):
    pairs, by_user = [], {}
    with path.open() as stream:
        for line in stream:
            values = [int(value) for value in line.split()]
            if len(values) < 2:
                continue
            user, items = values[0], list(dict.fromkeys(values[1:]))
            by_user[user] = items
            pairs.extend((user, item) for item in items)
    return np.asarray(pairs, dtype=np.int64), by_user


def load_dataset(root):
    train_pairs, train_users = read_split(root / "train.txt")
    test_pairs, test_users = read_split(root / "test.txt")
    valid_users, heldout_test = {}, {}
    for user, items in test_users.items():
        cut = max(1, len(items) // 2)
        valid_users[user] = items[:cut]
        heldout_test[user] = items[cut:] or items[-1:]
    test_users = heldout_test
    n_users = int(max(train_pairs[:, 0].max(), test_pairs[:, 0].max()) + 1)
    n_items = int(max(train_pairs[:, 1].max(), test_pairs[:, 1].max()) + 1)
    return train_pairs, train_users, valid_users, test_users, n_users, n_items


def normalized_graph(pairs, n_users, n_items, device):
    users = torch.from_numpy(pairs[:, 0])
    items = torch.from_numpy(pairs[:, 1]) + n_users
    rows, cols = torch.cat((users, items)), torch.cat((items, users))
    degree = torch.bincount(rows, minlength=n_users + n_items).float().clamp_min(1)
    values = degree[rows].rsqrt() * degree[cols].rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack((rows, cols)).to(device), values.to(device),
        (n_users + n_items,) * 2, device=device
    ).coalesce()


class CFModel(nn.Module):
    def __init__(self, n_users, n_items, dim, layers, graph):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.layers, self.graph = layers, graph
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def embeddings(self, graph=None):
        if self.layers == 0:
            return self.user_embedding.weight, self.item_embedding.weight
        graph = self.graph if graph is None else graph
        current = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        outputs = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(graph, current)
            outputs.append(current)
        final = torch.stack(outputs).mean(0)
        return final[:self.n_users], final[self.n_users:]


def sample_negatives(users, train_sets, n_items, rng):
    negatives = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    for index, user in enumerate(users):
        while int(negatives[index]) in train_sets[int(user)]:
            negatives[index] = rng.integers(0, n_items)
    return negatives


def bpr_loss(user_emb, positive_emb, negative_emb, l2):
    positive = (user_emb * positive_emb).sum(1)
    negative = (user_emb * negative_emb).sum(1)
    ranking = -F.logsigmoid(positive - negative).mean()
    regularizer = (user_emb.square().sum() + positive_emb.square().sum()
                   + negative_emb.square().sum()) / (2 * len(user_emb))
    return ranking + l2 * regularizer


def drop_graph(graph, dropout):
    keep = torch.rand(graph.values().shape, device=graph.device) >= dropout
    return torch.sparse_coo_tensor(
        graph.indices()[:, keep], graph.values()[keep] / (1 - dropout),
        graph.shape, device=graph.device
    ).coalesce()


def info_nce(first, second, temperature):
    first, second = F.normalize(first), F.normalize(second)
    logits = first @ second.t() / temperature
    return F.cross_entropy(logits, torch.arange(len(first), device=first.device))


@torch.no_grad()
def evaluate(model, train_users, test_users, topk, batch_size):
    model.eval()
    user_emb, item_emb = model.embeddings()
    users, recall_sum, ndcg_sum = sorted(test_users), 0.0, 0.0
    discounts = np.log2(np.arange(2, topk + 2))
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        scores = user_emb[batch_users] @ item_emb.t()
        for row, user in enumerate(batch_users):
            scores[row, list(train_users.get(user, ()))] = -torch.inf
        ranking = scores.topk(topk, dim=1).indices.cpu().numpy()
        for row, user in enumerate(batch_users):
            relevant = set(test_users[user])
            hits = np.asarray([item in relevant for item in ranking[row]], dtype=float)
            recall_sum += hits.sum() / len(relevant)
            ideal = min(len(relevant), topk)
            ndcg_sum += (hits / discounts).sum() / (1 / discounts[:ideal]).sum()
    count = len(users)
    return {"recall": recall_sum / count, "ndcg": ndcg_sum / count}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
    pairs, train_users, valid_users, test_users, n_users, n_items = load_dataset(Path(args.data_path) / args.dataset)
    train_sets = {user: set(items) for user, items in train_users.items()}
    graph = normalized_graph(pairs, n_users, n_items, device)
    layers = 0 if args.model == "bpr" else args.layers
    model = CFModel(n_users, n_items, args.dim, layers, graph).to(device)
    optimizer, rng = torch.optim.Adam(model.parameters(), lr=args.lr), np.random.default_rng(args.seed)
    best_valid, best_state, stale = {"recall": -1.0, "ndcg": -1.0, "epoch": -1}, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order, epoch_loss = rng.permutation(len(pairs)), 0.0
        for start in range(0, len(order), args.batch_size):
            batch = pairs[order[start:start + args.batch_size]]
            users_np, positives_np = batch[:, 0], batch[:, 1]
            negatives_np = sample_negatives(users_np, train_sets, n_items, rng)
            users = torch.from_numpy(users_np).to(device)
            positives = torch.from_numpy(positives_np).to(device)
            negatives = torch.from_numpy(negatives_np).to(device)
            all_users, all_items = model.embeddings()
            loss = bpr_loss(all_users[users], all_items[positives], all_items[negatives], args.l2)
            if args.model == "sgl":
                view1, view2 = model.embeddings(drop_graph(graph, args.dropout)), model.embeddings(drop_graph(graph, args.dropout))
                unique_users, unique_items = users.unique(), positives.unique()
                ssl = info_nce(view1[0][unique_users], view2[0][unique_users], args.temperature)
                ssl += info_nce(view1[1][unique_items], view2[1][unique_items], args.temperature)
                loss = loss + args.ssl_weight * ssl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        metrics = evaluate(model, train_users, valid_users, args.topk, args.eval_batch_size)
        print(f"epoch={epoch:03d} loss={epoch_loss / len(pairs):.6f} "
              f"recall@{args.topk}={metrics['recall']:.6f} ndcg@{args.topk}={metrics['ndcg']:.6f}", flush=True)
        if metrics["recall"] > best_valid["recall"]:
            best_valid = {**metrics, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.validation_only:
        print("VALIDATION_RESULT=" + json.dumps({
            "model": args.model, "dataset": args.dataset,
            "valid_recall": best_valid["recall"],
            "valid_ndcg": best_valid["ndcg"],
            "selected_epoch": best_valid["epoch"],
        }, sort_keys=True))
        return
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, train_users, test_users, args.topk, args.eval_batch_size)
    result = {"model": args.model, "dataset": args.dataset, "topk": args.topk,
              "selected_epoch": best_valid["epoch"],
              "valid_recall": best_valid["recall"], "valid_ndcg": best_valid["ndcg"],
              "recall": test_metrics["recall"], "ndcg": test_metrics["ndcg"]}
    (output / f"{args.dataset}_{args.model}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
