#!/usr/bin/env python3
"""ItemKNN on EditKG interactions and relation/entity content features."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from nsrs_protocol import protocol

# Reuse the tested sparse similarity/weighting implementation, not CAFE's data loader.
spec = importlib.util.spec_from_file_location('nsrs_itemknn_math',
    ROOT.parent / 'CAFE/baselines/models/itemknn_hybrids.py')
math = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math
spec.loader.exec_module(math)


def pairs(path):
    output = []
    for line in path.read_text().splitlines():
        values = list(map(int, line.split()))
        if values:
            output.extend((values[0], item) for item in set(values[1:]))
    return np.asarray(output, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    args = parser.parse_args()
    params = json.loads(args.params.read_text())
    source = args.data_root / args.dataset
    train_pairs, targets = pairs(source / 'train.txt'), pairs(source / 'test.txt')
    n_users = int(max(train_pairs[:, 0].max(), targets[:, 0].max())) + 1
    n_items = int(max(train_pairs[:, 1].max(), targets[:, 1].max())) + 1
    train = sp.csr_matrix((np.ones(len(train_pairs)), train_pairs.T), shape=(n_users, n_items), dtype=np.float32)
    content = None
    if args.model != 'itemknn-cf':
        rows, columns, vocabulary = [], [], {}
        for line in (source / 'kg_final.txt').open():
            head, relation, tail = map(int, line.split())
            for item, feature in ((head, ('out', relation, tail)), (tail, ('in', relation, head))):
                if item < n_items:
                    column = vocabulary.setdefault(feature, len(vocabulary))
                    rows.append(item)
                    columns.append(column)
        content = sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n_items, len(vocabulary)), dtype=np.float32)
        content.data[:] = 1.0
    name = {'itemknn-cf': 'ItemKNN-CF', 'itemknn-cbf': 'ItemKNN-CBF', 'itemknn-cfcbf': 'ItemKNN-CFCBF'}[args.model]
    profiles = math.build_item_profiles(name, train, content, params['interaction_weighting'],
                                       params['content_weighting'], params['content_weight'])
    similarity = math.topk_cosine_similarity(profiles, params['neighbors'], params['shrink'], params['normalize'], 256)
    by_user = {}
    for user, item in targets:
        by_user.setdefault(int(user), set()).add(int(item))
    users = sorted(by_user)
    recall, ndcg = [], []
    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        scores = (train[batch] @ similarity).toarray()
        for index, user in enumerate(batch):
            seen = train[user].indices
            scores[index, seen] = -np.inf
            ranking = np.argsort(-scores[index], kind='stable')[:min(20, n_items - len(seen))]
            hits = np.array([item in by_user[user] for item in ranking], dtype=float)
            discounts = np.log2(np.arange(2, len(ranking) + 2))
            recall.append(float(hits.sum() / len(by_user[user])))
            ideal = np.sum(1.0 / np.log2(np.arange(2, min(20, len(by_user[user])) + 2)))
            ndcg.append(float(np.sum(hits / discounts) / ideal))
    protocol.observe(1, {'recall': float(np.mean(recall)), 'ndcg': float(np.mean(ndcg))})


if __name__ == '__main__':
    main()
