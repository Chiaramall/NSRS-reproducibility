"""Model-specific search space and training arguments."""
from hyperopt import hp
from common import METRICS, trial_command

def search_space():
    return {
        'lr': hp.choice('lr', [1e-4, 5e-4, 1e-3, 5e-3]),
        'l2': hp.choice('l2', [1e-6, 1e-5, 1e-4, 1e-3]),
        'dropout': hp.choice('dropout', [0.0, 0.1, 0.2, 0.3, 0.5]),
        'batch_size': hp.choice('batch_size', [64, 128, 256, 512]),
        'hidden_size': hp.choice('hidden_size', [32, 64, 128, 256]),
        'num_layers': hp.choice('num_layers', [1, 2]),
        'output_size': hp.choice('output_size', [32, 64, 128, 256]),
        'negative_mode': hp.choice('negative_mode', [
            'separate_embedding', 'sign_inversion', 'mlp'
        ]),
    }


def normalized_params(params):
    params = dict(params)
    if params['negative_mode'] == 'separate_embedding':
        params['neg_emb'] = 1
        params['neg_layer'] = '[]'
    elif params['negative_mode'] == 'mlp':
        params['neg_emb'] = 0
        params['neg_layer'] = '[%d]' % int(params['output_size'])
    else:
        params['neg_emb'] = 0
        params['neg_layer'] = '[]'
    params['p_layers'] = '[%d]' % int(params['output_size'])
    return params


def trial_command(args, params, trial_dir):
    command = [
        args.python, 'main.py',
        '--rank', '1', '--model_name', MODEL,
        '--optimizer', 'Adam', '--lr', str(params['lr']),
        '--dataset', args.dataset, '--metric', METRICS,
        '--test_neg_n', '100', '--max_his', '5',
        '--batch_size', str(params['batch_size']),
        '--eval_batch_size', str(args.eval_batch_size),
        '--l2', str(params['l2']), '--dropout', str(params['dropout']),
        '--epoch', str(args.epoch), '--early_stop', '1', '--check_epoch', '5',
        '--random_seed', str(args.training_seed), '--gpu', str(args.gpu),
        '--log_file', str(trial_dir / 'run.log'),
        '--result_file', str(trial_dir / 'result.npy'),
        '--model_path', str(trial_dir / 'model.pt'),
        '--u_vector_size', str(params['output_size']),
        '--i_vector_size', str(params['output_size']),
        '--hidden_size', str(params['hidden_size']),
        '--num_layers', str(params['num_layers']),
        '--p_layers', params['p_layers'],
        '--neg_emb', str(params['neg_emb']),
        '--neg_layer', params['neg_layer'],
    ]
    return command


MODEL = 'GRU4Rec'
