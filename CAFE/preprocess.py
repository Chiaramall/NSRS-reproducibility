from __future__ import absolute_import, division, print_function

import os
import pickle

import numpy as np
from tqdm import tqdm

from my_knowledge_graph import *
import utils


def load_kg_embedding(dataset: str):
    """Note that entity embedding is of size [vocab_size+1, d]."""
    print('>>> Load KG embeddings ...')
    state_dict = utils.load_embed_sd(dataset)
    # print(state_dict.keys())
    embeds = dict()
    # Load entity embeddings
    for entity in [USER, PRODUCT, WORD, BRAND, CATEGORY, RPRODUCT]:
        embeds[entity] = state_dict[entity + '.weight'].cpu().data.numpy()[:-1]   # remove last dummy embed with 0 values.
        print(f'>>> {entity}: {embeds[entity].shape}')
    for rel in [PURCHASE, MENTION, DESCRIBED_BY, PRODUCED_BY, BELONG_TO, ALSO_BUY, ALSO_VIEW, BUY_TOGETHER]:
        if rel == DESCRIBED_BY:
            embeds[rel] = (
                state_dict['describe_as'].cpu().data.numpy()[0],
                state_dict['describe_as_bias.weight'].cpu().data.numpy()
            )
        else:
            embeds[rel] = (
                state_dict[rel].cpu().data.numpy()[0],
                state_dict[rel + '_bias.weight'].cpu().data.numpy()
            )
    return embeds


def compute_top100_items(dataset, user_batch_size=256):
    """Compute the exact upstream ranking without materializing all user-item scores."""
    embeds = utils.load_embed(dataset)
    user_embed = embeds[USER]
    product_embed = embeds[PRODUCT]
    purchase_embed, purchase_bias = embeds[PURCHASE]
    del purchase_bias  # Kept for parity: the released implementation does not use it.
    num_users, num_products = user_embed.shape[0], product_embed.shape[0]
    k = min(100, num_products)
    best100 = np.empty((num_users, k), dtype=np.int64)
    product_t = product_embed.T
    pbar = tqdm(total=num_users)
    for start in range(0, num_users, user_batch_size):
        stop = min(start + user_batch_size, num_users)
        scores = np.dot(user_embed[start:stop] + purchase_embed, product_t)
        candidates = np.argpartition(scores, num_products - k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        best100[start:stop] = np.take_along_axis(candidates, order, axis=1)
        pbar.update(stop - start)
    pbar.close()
    print(best100.shape)
    return best100


def _save_partial(path, value):
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def estimate_path_count(args, checkpoint_every=250):
    kg = utils.load_kg(args.dataset)
    num_mp = len(kg.metapaths)
    train_labels = utils.load_labels(args.dataset, 'train')
    partial_file = utils.TMP_DIR[args.dataset] + "/path_count.partial.pkl"
    counts = {}
    if os.path.exists(partial_file):
        try:
            with open(partial_file, "rb") as handle:
                counts = pickle.load(handle)
            print(f">>> Resume path counts after {len(counts)} users.")
        except (EOFError, OSError, pickle.UnpicklingError):
            counts = {}
    pbar = tqdm(total=len(train_labels), initial=len(counts))
    since_checkpoint = 0
    for uid in train_labels:
        if uid in counts:
            continue
        counts[uid] = np.zeros(num_mp)
        for pid in train_labels[uid]:
            for mpid in range(num_mp):
                cnt = kg.count_paths_with_target(mpid, uid, pid, 50)
                counts[uid][mpid] += cnt
        counts[uid] = counts[uid] / len(train_labels[uid])
        pbar.update(1)
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            _save_partial(partial_file, counts)
            since_checkpoint = 0
    pbar.close()
    utils.save_path_count(args.dataset, counts)
    if os.path.exists(partial_file):
        os.unlink(partial_file)


def main(args):
    runtime_dir = utils.TMP_DIR[args.dataset]
    os.makedirs(runtime_dir, exist_ok=True)
    embed_file = runtime_dir + "/embed.pkl"
    kg_file = runtime_dir + "/kg.pkl"
    products_file = runtime_dir + "/user_products_pos.npy"
    count_file = runtime_dir + "/path_count.pkl"

    if os.path.exists(embed_file):
        print(f">>> Found {embed_file}; skipping embedding extraction.")
    else:
        embeds = load_kg_embedding(args.dataset)
        utils.save_embed(args.dataset, embeds)

    if os.path.exists(kg_file):
        print(f">>> Found {kg_file}; skipping KG construction.")
    else:
        kg = MyKnowledgeGraph(args.dataset)
        utils.save_kg(args.dataset, kg)

    if os.path.exists(products_file):
        print(f">>> Found {products_file}; skipping top-100 computation.")
    else:
        best100 = compute_top100_items(args.dataset)
        utils.save_user_products(args.dataset, best100, 'pos')

    if os.path.exists(count_file):
        print(f">>> Found {count_file}; skipping path-count estimation.")
    else:
        estimate_path_count(args)


if __name__ == '__main__':
    args = utils.parse_args()
    main(args)
