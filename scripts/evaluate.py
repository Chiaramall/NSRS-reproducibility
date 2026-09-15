# coding=utf-8
"""Run validation-NDCG@5-selected configs over multiple seeds."""

from __future__ import print_function

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_best_metrics
import tune


METRIC_NAMES = ['NDCG@5', 'NDCG@10', 'HR@5', 'HR@10']


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Run all NDCG@5-selected best configs over seed grid.')
    parser.add_argument('--models', default='all')
    parser.add_argument('--best_dir', type=Path,
                        default=repo_root / 'tuning' / 'hyperopt_ndcg5_ml100k')
    parser.add_argument('--output_dir', type=Path,
                        default=repo_root / 'tuning' / 'ndcg5_best_seed_grid_ml100k')
    parser.add_argument('--seeds', default='2019,2020,2021,2022,2023')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='ml100k01-1-5')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=16384)
    return parser.parse_args()


def parse_seeds(raw_seeds):
    return [int(seed.strip()) for seed in raw_seeds.split(',') if seed.strip()]


def metrics_if_complete(run_dir):
    metrics_file = run_dir / 'metrics.json'
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    log_file = run_dir / 'run.log'
    if not log_file.exists():
        return None
    try:
        metrics = read_best_metrics(log_file)
    except RuntimeError:
        return None
    metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def run_seed(args, model, params, seed, model_dir):
    run_dir = model_dir / ('seed_%d' % seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = metrics_if_complete(run_dir)
    if existing is not None:
        print('%s seed=%d already complete' % (model, seed), flush=True)
        return existing

    command_args = argparse.Namespace(
        python=args.python,
        dataset=args.dataset,
        eval_batch_size=args.eval_batch_size,
        epoch=args.epoch,
        training_seed=seed,
        gpu=args.gpu,
    )
    command = tune.command_for_model(command_args, model, params, run_dir)
    (run_dir / 'command.json').write_text(json.dumps(command, indent=2))
    (run_dir / 'params.json').write_text(
        json.dumps(params, indent=2, sort_keys=True))

    print('START %s seed=%d' % (model, seed), flush=True)
    import subprocess
    src_dir = Path(__file__).resolve().parents[1] / 'src'
    with (run_dir / 'stdout.log').open('w') as stdout_file:
        completed = subprocess.run(
            command, cwd=str(src_dir), stdout=stdout_file,
            stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(
            '%s seed=%d failed with exit code %d; see %s' %
            (model, seed, completed.returncode, run_dir / 'stdout.log'))

    metrics = read_best_metrics(run_dir / 'run.log')
    (run_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True))
    print('END %s seed=%d test=%s' %
          (model, seed, ','.join('%.4f' % x for x in metrics['test'])),
          flush=True)
    return metrics


def write_model_summary(model_dir, rows):
    validation = np.array([row['validation'] for row in rows], dtype=float)
    test = np.array([row['test'] for row in rows], dtype=float)
    summary = {
        'seeds': [row['seed'] for row in rows],
        'validation_mean': validation.mean(axis=0).round(6).tolist(),
        'validation_std': validation.std(axis=0, ddof=1).round(6).tolist(),
        'test_mean': test.mean(axis=0).round(6).tolist(),
        'test_std': test.std(axis=0, ddof=1).round(6).tolist(),
    }
    (model_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_csv(output_dir, summaries):
    with (output_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'metric', 'test_mean', 'test_std',
                         'validation_mean', 'validation_std'])
        for model, summary in summaries.items():
            for idx, metric in enumerate(METRIC_NAMES):
                writer.writerow([
                    model,
                    metric,
                    summary['test_mean'][idx],
                    summary['test_std'][idx],
                    summary['validation_mean'][idx],
                    summary['validation_std'][idx],
                ])


def main():
    args = parse_args()
    args.best_dir = args.best_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for model in tune.selected_models(args.models):
        model_dir_name = tune.MODEL_DIRS[model]
        best_file = args.best_dir / model_dir_name / 'best.json'
        best = json.loads(best_file.read_text())
        params = best['params']
        model_dir = args.output_dir / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / 'selected_best.json').write_text(
            json.dumps(best, indent=2, sort_keys=True))

        rows = []
        for seed in parse_seeds(args.seeds):
            metrics = run_seed(args, model, params, seed, model_dir)
            rows.append({
                'seed': seed,
                'best_epoch': metrics['epoch'],
                'validation': metrics['validation'],
                'test': metrics['test'],
            })
        (model_dir / 'seed_metrics.json').write_text(
            json.dumps(rows, indent=2, sort_keys=True))
        summaries[model] = write_model_summary(model_dir, rows)
        print('%s test mean=%s std=%s' % (
            model,
            ','.join('%.4f' % x for x in summaries[model]['test_mean']),
            ','.join('%.4f' % x for x in summaries[model]['test_std'])),
            flush=True)

    write_csv(args.output_dir, summaries)


if __name__ == '__main__':
    main()
