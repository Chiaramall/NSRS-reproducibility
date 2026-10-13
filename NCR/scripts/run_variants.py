#!/usr/bin/env python
"""Train/evaluate the three NCR variants using the recorded configurations."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from common import METRICS, read_best_metrics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ml100k01-1-5', '5MoviesTV01-1-5', 'Electronics01-1-5')
MODELS = ('NCR', 'NCR-E', 'NCR-E-WoLR')


def parameters(model, dataset):
    recorded = json.loads((ROOT.parent / 'hyperparameters/NCR.json').read_text())
    # Explicit shared settings fill fields omitted from the recorded best entry.
    params = dict(optimizer='Adam', lr=0.001, l2=0.0001, batch_size=128,
                  dropout=0.2, u_vector_size=64, i_vector_size=64,
                  r_weight=0.0 if model == 'NCR-E-WoLR' else 0.1,
                  ppl_weight=0, pos_weight=0, max_his=5)
    params.update(recorded['models'][model]['best_hyperparameters'][dataset])
    return params


def command_for(args, model, seed, run_dir):
    command = [sys.executable, 'main.py', '--rank', '1', '--model_name', model,
               '--dataset', args.dataset, '--metric', METRICS,
               '--test_neg_n', '100', '--random_seed', str(seed),
               '--gpu', args.gpu, '--epoch', str(args.epoch),
               '--early_stop', '1', '--check_epoch', '5',
               '--log_file', str(run_dir / 'run.log'),
               '--result_file', str(run_dir / 'result.npy'),
               '--model_path', str(run_dir / 'model.pt')]
    for key, value in parameters(model, args.dataset).items():
        command.extend(['--' + key, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', default='2019,2020,2021,2022,2023')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tuning/variants')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or args.epoch < 1:
        parser.error('Use unique seeds and a positive epoch budget')
    for model in args.models:
        rows = []
        model_dir = args.output_dir.resolve() / args.dataset / model
        for seed in seeds:
            run_dir = model_dir / ('seed_%s' % seed)
            command = command_for(args, model, seed, run_dir)
            print(' '.join(shlex.quote(part) for part in command), flush=True)
            if args.dry_run:
                continue
            identity = dict(command=command, parameters=parameters(model, args.dataset))
            metadata = run_dir / 'command.json'
            metrics_file = run_dir / 'metrics.json'
            if metadata.exists() and json.loads(metadata.read_text()) != identity:
                raise RuntimeError('Configuration changed; use a new --output-dir: %s' % run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(identity, indent=2) + '\n')
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
            else:
                with (run_dir / 'stdout.log').open('w') as output:
                    subprocess.run(command, cwd=str(ROOT / 'src'), stdout=output,
                                   stderr=subprocess.STDOUT, check=True)
                metrics = read_best_metrics(run_dir / 'run.log')
                metrics_file.write_text(json.dumps(metrics, indent=2) + '\n')
            rows.append(dict(seed=seed, **metrics))
        if rows:
            summary = dict(model=model, dataset=args.dataset, per_seed=rows,
                           metric_order=METRICS.split(','),
                           parameters=parameters(model, args.dataset),
                           selection='epoch selected by validation NDCG@5; fixed recorded hyperparameters')
            for split in ('validation', 'test'):
                summary[split + '_mean'] = [statistics.mean(row[split][i] for row in rows) for i in range(4)]
                summary[split + '_std'] = [statistics.stdev(row[split][i] for row in rows) if len(rows) > 1 else 0.0 for i in range(4)]
            (model_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
