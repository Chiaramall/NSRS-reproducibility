# coding=utf-8
"""Hyperopt/TPE search for all NCR baselines maximizing validation NDCG@5."""

from __future__ import print_function

import argparse
from functools import partial
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe
    from hyperopt.exceptions import AllTrialsFailed
except ImportError:
    raise SystemExit(
        'Hyperopt is required. Install it with: '
        'python -m pip install -r requirements-hyperopt.txt')

from common import read_best_metrics
import baseline_config as base_tune
import hidasi_config as hyperopt_hidasi
import transformer_config as hyperopt_sas_bert
import gru_config as hyperopt_gru4rec


OBJECTIVE_INDEX = 0
OBJECTIVE_NAME = 'validation_ndcg_at_5'
BASE_MODELS = ('BPR-MF', 'SVD++', 'DMF', 'NeuMF', 'STAMP')
ALL_MODELS = BASE_MODELS + (
    'GRU4Rec', 'GRU4RecHidasi', 'SASRec', 'BERT4Rec')
MODEL_DIRS = {
    'BPR-MF': 'BPR-MF',
    'SVD++': 'SVDplusplus',
    'DMF': 'DMF',
    'NeuMF': 'NeuMF',
    'STAMP': 'STAMP',
    'GRU4Rec': 'GRU4Rec',
    'GRU4RecHidasi': 'GRU4RecHidasi',
    'SASRec': 'SASRec',
    'BERT4Rec': 'BERT4Rec',
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Hyperopt/TPE search maximizing validation NDCG@5.')
    parser.add_argument('--models', default='all')
    parser.add_argument('--max_evals', type=int, default=30)
    parser.add_argument('--startup_jobs', type=int, default=8)
    parser.add_argument('--training_seed', type=int, default=2021)
    parser.add_argument('--sampler_seed', type=int, default=2026)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset', default='ml100k01-1-5')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=16384)
    parser.add_argument('--output_dir', type=Path,
                        default=repo_root / 'tuning' / 'hyperopt_ndcg5_ml100k')
    return parser.parse_args()


def selected_models(raw_models):
    if raw_models.strip().lower() == 'all':
        return list(ALL_MODELS)
    models = [model.strip() for model in raw_models.split(',') if model.strip()]
    unknown = sorted(set(models) - set(ALL_MODELS))
    if unknown:
        raise ValueError('Unsupported models: %s' % ', '.join(unknown))
    return models


def load_state(state_file, sampler_seed):
    if not state_file.exists():
        return Trials(), np.random.default_rng(sampler_seed)
    with state_file.open('rb') as state_handle:
        state = pickle.load(state_handle)
    random_state = np.random.default_rng()
    random_state.bit_generator.state = state['random_state']
    return state['trials'], random_state


def save_state(state_file, trials, random_state):
    temporary = state_file.with_suffix('.tmp')
    with temporary.open('wb') as state_handle:
        pickle.dump({
            'trials': trials,
            'random_state': random_state.bit_generator.state,
        }, state_handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(state_file)


def successful_results(trials):
    return [
        trial['result'] for trial in trials.trials
        if trial['result'].get('status') == STATUS_OK
    ]


def save_summary(model_dir, trials):
    results = successful_results(trials)
    if not results:
        return
    results.sort(key=lambda result: result['loss'])
    (model_dir / 'trials.json').write_text(
        json.dumps(results, indent=2, sort_keys=True))
    best = dict(results[0])
    best[OBJECTIVE_NAME] = -best.pop('loss')
    (model_dir / 'best.json').write_text(
        json.dumps(best, indent=2, sort_keys=True))


def next_trial_number(model_dir):
    numbers = []
    for path in model_dir.glob('trial_*'):
        try:
            numbers.append(int(path.name.split('_')[-1]))
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 0


def model_space(model):
    if model in BASE_MODELS:
        return base_tune.search_space(model)
    if model == 'GRU4RecHidasi':
        return hyperopt_hidasi.search_space()
    if model in ('SASRec', 'BERT4Rec'):
        return hyperopt_sas_bert.search_space(model)
    if model == 'GRU4Rec':
        return hyperopt_gru4rec.search_space()
    raise ValueError('Unsupported model: %s' % model)


def normalize_params(model, params):
    if model == 'GRU4RecHidasi':
        return hyperopt_hidasi.normalized_params(params)
    if model in ('SASRec', 'BERT4Rec'):
        return hyperopt_sas_bert.normalized_params(params)
    if model == 'GRU4Rec':
        return hyperopt_gru4rec.normalized_params(params)
    return dict(params)


def command_for_model(args, model, params, trial_dir):
    if model in BASE_MODELS:
        return base_tune.trial_command(args, model, params, trial_dir)
    if model == 'GRU4RecHidasi':
        return hyperopt_hidasi.trial_command(args, params, trial_dir)
    if model in ('SASRec', 'BERT4Rec'):
        return hyperopt_sas_bert.trial_command(args, model, params, trial_dir)
    if model == 'GRU4Rec':
        return hyperopt_gru4rec.trial_command(args, params, trial_dir)
    raise ValueError('Unsupported model: %s' % model)


def make_objective(args, model, model_dir, counter):
    src_dir = Path(__file__).resolve().parents[1] / 'src'

    def objective(raw_params):
        params = normalize_params(model, raw_params)
        trial_number = counter[0]
        counter[0] += 1
        trial_dir = model_dir / ('trial_%04d' % trial_number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        command = command_for_model(args, model, params, trial_dir)
        (trial_dir / 'params.json').write_text(
            json.dumps(params, indent=2, sort_keys=True))
        (trial_dir / 'command.json').write_text(json.dumps(command, indent=2))

        import subprocess
        with (trial_dir / 'stdout.log').open('w') as stdout_file:
            completed = subprocess.run(
                command, cwd=str(src_dir), stdout=stdout_file,
                stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            return {
                'status': STATUS_FAIL,
                'loss': 1e9,
                'params': params,
                'trial_dir': str(trial_dir),
                'error': 'exit_code_%d' % completed.returncode,
            }

        try:
            metrics = read_best_metrics(trial_dir / 'run.log')
        except RuntimeError as error:
            return {
                'status': STATUS_FAIL,
                'loss': 1e9,
                'params': params,
                'trial_dir': str(trial_dir),
                'error': str(error),
            }

        return {
            'status': STATUS_OK,
            'loss': -metrics['validation'][OBJECTIVE_INDEX],
            'params': params,
            'best_epoch': metrics['epoch'],
            'validation_metrics': metrics['validation'],
            'test_metrics': metrics['test'],
            'trial_dir': str(trial_dir),
        }

    return objective


def tune_model(args, model):
    model_dir = args.output_dir / MODEL_DIRS[model]
    model_dir.mkdir(parents=True, exist_ok=True)
    state_file = model_dir / 'hyperopt_state.pkl'
    trials, random_state = load_state(state_file, args.sampler_seed)
    if len(trials.trials) >= args.max_evals:
        print('%s already has %d evaluations' % (model, len(trials.trials)))
        save_summary(model_dir, trials)
        return

    counter = [next_trial_number(model_dir)]
    objective = make_objective(args, model, model_dir, counter)
    algorithm = partial(tpe.suggest, n_startup_jobs=args.startup_jobs)
    while len(trials.trials) < args.max_evals:
        target = len(trials.trials) + 1
        print('%s Hyperopt NDCG@5 evaluation %d/%d' %
              (model, target, args.max_evals), flush=True)
        try:
            fmin(
                fn=objective,
                space=model_space(model),
                algo=algorithm,
                max_evals=target,
                trials=trials,
                rstate=random_state,
                show_progressbar=False,
                verbose=False,
            )
        except AllTrialsFailed:
            print('%s has no successful evaluations yet; continuing' % model,
                  flush=True)
        save_state(state_file, trials, random_state)
        save_summary(model_dir, trials)

    results = successful_results(trials)
    if not results:
        print('%s finished without a successful evaluation' % model,
              flush=True)
        return
    best = min(results, key=lambda result: result['loss'])
    print('%s best validation NDCG@5 %.6f' % (model, -best['loss']))
    print('Parameters: %s' % best['params'])


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in selected_models(args.models):
        tune_model(args, model)


if __name__ == '__main__':
    main()
