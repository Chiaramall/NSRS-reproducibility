#!/usr/bin/env python3
"""Validation search, fixed-duration refit, and multi-seed final evaluation."""
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('last-fm', 'amazon-book', 'yelp2018')
SIMPLE = ('bpr', 'lightgcn', 'sgl')
RECBOLE = ('gcmc', 'cke', 'ktup', 'kgcn')
NATIVE = ('editkg', 'kgat', 'kgin', 'kgcl', 'kgrec', 'krdn')
KNN = ('itemknn-cf', 'itemknn-cbf', 'itemknn-cfcbf')
MODELS = NATIVE + SIMPLE + RECBOLE + KNN


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def candidates(archive, model, dataset, count, seed):
    record = next(value for key, value in archive['models'].items() if key.lower() == model)
    optimization = record['optimization']
    space = optimization.get('recorded_search_space', optimization.get('search_space'))
    if 'configurations' in space:
        columns = space.get('columns')
        choices = [dict(zip(columns, row)) if columns else row for row in space['configurations']]
        return choices[:count]
    space = {key: list(values) for key, values in space.items()}
    for key, values in optimization.get('author_confirmed_values', {}).get(dataset, {}).items():
        space[key] = list(dict.fromkeys(space.get(key, []) + values))
    rng = random.Random('%s:%s:%s' % (seed, model, dataset))
    result, seen = [], set()
    # Explicit baseline candidate screens are tried before seeded random candidates.
    screen = optimization.get('searches', {}).get('paper_hparams_tuning', {}).get('search_space', {}).get('configurations', [])
    for candidate in screen:
        token = digest(candidate)
        if token not in seen:
            result.append(candidate); seen.add(token)
    attempts = 0
    while len(result) < count and attempts < count * 100:
        candidate = {key: rng.choice(values) for key, values in sorted(space.items())}
        token = digest(candidate)
        if token not in seen:
            result.append(candidate); seen.add(token)
        attempts += 1
    return result[:count]


def native_parameters(model, params):
    params = dict(params)
    if model == 'kgat':
        if 'reg' in params:
            params['regs'] = [params.pop('reg')] * 2
        if 'dropout' in params:
            value = params.pop('dropout')
            params['node_dropout'] = [value]
            params['mess_dropout'] = [value] * len(ast.literal_eval(params['layer_size']))
    return params


def command(args, model, dataset, phase, params, epochs, seed, work):
    data_root = args.validation_root if phase == 'validation' else args.data_root
    env = dict(os.environ, NSRS_PHASE=phase, NSRS_RESULT_FILE=str(work / 'metrics.json'),
               NSRS_SEED=str(seed), EDITKG_DATA_ROOT=str(data_root),
               EDITKG_PYTHON=args.python, KGAT_PYTHON=args.kgat_python,
               GPU_ID=args.gpu, CUDA_VISIBLE_DEVICES='' if args.cpu else args.gpu,
               EDITKG_EVAL_STEP='1', EDITKG_STOPPING_STEP=str(epochs + 1),
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    if model in RECBOLE or model in KNN:
        runner = 'run_recbole_protocol.py' if model in RECBOLE else 'run_itemknn_protocol.py'
        cmd = [args.python, str(ROOT / 'scripts' / runner), '--model', model, '--dataset', dataset,
               '--data-root', str(data_root), '--params', str(work / 'params.json')]
        if model in RECBOLE:
            cmd += ['--work-dir', str(work), '--epochs', str(epochs), '--seed', str(seed)]
            if args.cpu: cmd.append('--cpu')
    else:
        cmd = ['bash', str(ROOT / 'scripts/run_baseline.sh'), model, dataset]
        for key, value in native_parameters(model, params).items():
            cmd += ['--' + key, json.dumps(value) if isinstance(value, list) else str(value)]
        cmd += ['--epochs' if model in SIMPLE else '--epoch', str(epochs)]
        if model in SIMPLE or model == 'editkg':cmd += ['--seed', str(seed)]
        if model in SIMPLE:
            cmd += ['--output', str(work)]
            if args.cpu:cmd += ['--cpu']
        elif args.cpu and model != 'kgat':
            cmd += ['--cuda', '0' if model in ('kgcl', 'kgrec') else 'false']
    return cmd, env


def execute(args, model, dataset, phase, params, epochs, seed, work):
    cmd, env = command(args, model, dataset, phase, params, epochs, seed, work)
    identity = dict(model=model, dataset=dataset, phase=phase, params=params, epochs=epochs,
                    seed=seed, command=cmd, protocol=args.identity)
    completed = work / 'complete.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved['identity'] != identity:
            raise RuntimeError('Incompatible saved run; use a new output directory: ' + str(work))
        return saved['metrics']
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(work / 'params.json', params)
    atomic_json(work / 'run.json', identity)
    (work / 'metrics.json').unlink(missing_ok=True)
    print('[%s] %s/%s: %s' % (phase, dataset, model, shlex.join(cmd)), flush=True)
    with (work / 'stdout.log').open('w') as log:
        process = subprocess.run(cmd, cwd=str(work), env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('Training failed; see ' + str(work / 'stdout.log'))
    metrics = json.loads((work / 'metrics.json').read_text())
    if metrics['phase'] != phase or metrics['last_epoch'] != epochs:
        raise RuntimeError('Run did not complete the requested epoch budget: ' + str(work))
    if phase == 'final' and metrics['evaluated_epochs'] != [epochs]:
        raise RuntimeError('Final test was consulted before the final epoch')
    atomic_json(completed, dict(identity=identity, metrics=metrics))
    return metrics


def check_splits(source, validation):
    def read(path):
        rows = {}
        with path.open() as stream:
            for line in stream:
                values = list(map(int, line.split()))
                if values: rows.setdefault(values[0], set()).update(values[1:])
        return rows
    full = read(source / 'train.txt')
    reduced = read(validation / 'train.txt')
    heldout = read(validation / 'test.txt')
    for user in full.keys() | reduced.keys() | heldout.keys():
        train, valid = reduced.get(user, set()), heldout.get(user, set())
        if train & valid or train | valid != full.get(user, set()):
            raise ValueError('Validation is not a disjoint partition of full training')
    if not any(heldout.values()):raise ValueError('Empty validation split')
    test = read(source / 'test.txt')
    if not any(test.values()):raise ValueError('Empty test split')
    for user, items in test.items():
        if items & full.get(user, set()):raise ValueError('Training/test overlap')
    if file_digest(source / 'kg_final.txt') != file_digest(validation / 'kg_final.txt'):
        raise ValueError('Training and validation KG differ')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'search', 'final', 'report', 'plan'), default='all')
    parser.add_argument('--models', nargs='+', choices=('all',) + MODELS, default=['all'])
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--search-seed', type=int, default=2026)
    parser.add_argument('--training-seed', type=int, default=2024)
    parser.add_argument('--final-seeds', nargs='+', type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--kgat-python', default=os.environ.get('KGAT_PYTHON', str(ROOT / '.envs/kgat-legacy/bin/python')))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/editkg-filtered')
    parser.add_argument('--validation-root', type=Path, default=ROOT / 'data/editkg-validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/benchmark')
    args = parser.parse_args()
    if args.trials < 1 or args.epochs < 1:parser.error('trials and epochs must be positive')
    if len(set(args.final_seeds)) != len(args.final_seeds):parser.error('final seeds must be unique')
    if args.models == ['all']: args.models = list(MODELS)
    elif 'all' in args.models: parser.error('Use all alone')
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        parser.error('models and datasets must not contain duplicates')
    for attr in ('data_root', 'validation_root', 'output'):
        setattr(args, attr, getattr(args, attr).resolve())
    return args


def main():
    args = parse_args()
    archive = json.loads((ROOT.parent / 'hyperparameters/EditKG.json').read_text())
    plans = {(dataset, model): candidates(archive, model, dataset, args.trials, args.search_seed)
             for dataset in args.datasets for model in args.models}
    if args.stage == 'plan':
        print(json.dumps([dict(dataset=d, model=m, candidates=len(c), max_epochs=args.epochs,
                               final_seeds=[args.final_seeds[0]] if m in KNN else args.final_seeds)
                          for (d, m), c in plans.items()], indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.benchmark.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another benchmark is using this output directory')
    if 'kgat' in args.models and args.stage in ('all', 'search', 'final') and not Path(args.kgat_python).is_file():
        raise FileNotFoundError('Set --kgat-python / KGAT_PYTHON to the Python 3.7 TensorFlow environment')
    # The manifest protects resumptions against changed data, source, or settings.
    code = [ROOT / 'nsrs_protocol.py', ROOT / 'main.py', ROOT / 'requirements.txt']
    for directory in ('scripts', 'modules', 'utils', 'baselines'):
        code.extend(p for p in (ROOT / directory).rglob('*.py')
                    if not any(part.startswith('.') for part in p.relative_to(ROOT).parts))
    code.extend((ROOT / 'configs').glob('*'))
    code.append(ROOT / 'scripts/run_baseline.sh')
    code_identity = digest({str(p.relative_to(ROOT)): file_digest(p) for p in code if p.is_file()})
    code_identity = digest([code_identity, file_digest(ROOT.parent / 'CAFE/baselines/models/itemknn_hybrids.py')])
    report = []
    for dataset in args.datasets:
        check_splits(args.data_root / dataset, args.validation_root / dataset)
        data_identity = {phase + '/' + name: file_digest(root / dataset / name)
                         for phase, root in [('full', args.data_root), ('validation', args.validation_root)]
                         for name in ('train.txt', 'test.txt', 'kg_final.txt')}
        for model in args.models:
            base = args.output / dataset / model
            identity = dict(version=1, model=model, dataset=dataset, code=code_identity,
                            data=data_identity, candidates=plans[dataset, model], max_epochs=args.epochs,
                            search_seed=args.search_seed, training_seed=args.training_seed,
                            final_seeds=args.final_seeds, cpu=args.cpu,
                            method='explicit candidate screens then seeded categorical random search',
                            objective='validation Recall@20, tie-break validation NDCG@20, earliest epoch')
            manifest = base / 'protocol.json'
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise RuntimeError('Protocol changed; use a new --output directory: ' + str(base))
            atomic_json(manifest, identity)
            args.identity = digest(identity)
            selection = base / 'selection.json'
            if args.stage in ('all', 'search'):
                rows = []
                for index, params in enumerate(plans[dataset, model]):
                    metrics = execute(args, model, dataset, 'validation', params,
                                      1 if model in KNN else args.epochs, args.training_seed,
                                      base / 'search' / ('trial_%03d' % index))
                    rows.append(dict(trial=index, params=params, metrics=metrics))
                chosen = max(rows, key=lambda row: (row['metrics']['recall@20'], row['metrics']['ndcg@20'], -row['metrics']['selected_epoch'], -row['trial']))
                atomic_json(selection, dict(chosen, protocol=args.identity))
            if args.stage in ('all', 'final', 'report'):
                selected = json.loads(selection.read_text())
                if selected['protocol'] != args.identity:raise RuntimeError('Selection protocol mismatch')
                results = []
                seeds = args.final_seeds[:1] if model in KNN else args.final_seeds
                for seed in seeds:
                    work = base / 'final' / ('seed_%s' % seed)
                    if args.stage == 'report':
                        saved = json.loads((work / 'complete.json').read_text())
                        if saved['identity']['protocol'] != args.identity:raise RuntimeError('Final result protocol mismatch')
                        metrics = saved['metrics']
                    else:
                        metrics = execute(args, model, dataset, 'final', selected['params'],
                                          selected['metrics']['selected_epoch'], seed, work)
                    results.append(dict(seed=seed, **metrics))
                summary = dict(model=model, dataset=dataset, selected=selected, per_seed=results,
                               deterministic=model in KNN)
                for metric in ('recall@20', 'ndcg@20'):
                    values = [row[metric] for row in results]
                    summary[metric + '_mean'] = statistics.mean(values)
                    summary[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                atomic_json(base / 'summary.json', summary)
                report.append({key: summary[key] for key in ('model', 'dataset', 'recall@20_mean', 'recall@20_std', 'ndcg@20_mean', 'ndcg@20_std')})
    if report:
        # A report is scoped to the requested subset; it does not overwrite other summaries.
        name = 'report_' + digest(dict(models=args.models, datasets=args.datasets))[:12] + '.csv'
        with (args.output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[0]))
            writer.writeheader(); writer.writerows(report)


if __name__ == '__main__':
    main()
