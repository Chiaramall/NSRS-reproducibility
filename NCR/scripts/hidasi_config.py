"""Model-specific search space and training arguments."""
from hyperopt import hp
from common import METRICS, trial_command

def search_space():
    return {
        'lr': hp.choice('lr', [0.005, 0.01, 0.025, 0.05, 0.1]),
        'batch_size': hp.choice('batch_size', [64, 128, 256, 512]),
        'layers': hp.choice('layers', ['[64]', '[100]', '[128]', '[64,64]']),
        'loss': hp.choice('loss', ['bpr-max', 'cross-entropy']),
        'dropout_embed': hp.choice(
            'dropout_embed', [0.0, 0.1, 0.2, 0.3, 0.5]),
        'dropout_hidden': hp.choice(
            'dropout_hidden', [0.0, 0.1, 0.2, 0.3, 0.5]),
        'momentum': hp.choice('momentum', [0.0, 0.1, 0.2]),
        'bpreg': hp.choice('bpreg', [0.1, 0.5, 1.0, 2.0, 10.0]),
        'elu_param': hp.choice('elu_param', [0.5, 1.0]),
        'adagrad_eps': hp.choice('adagrad_eps', [1e-8, 1e-6, 1e-4]),
        'constrained_embedding': hp.choice('constrained_embedding', [1, 0]),
        'embedding_size': hp.choice('embedding_size', [32, 64, 100, 128]),
    }


def normalized_params(params):
    params = dict(params)
    if int(params['constrained_embedding']) == 1:
        params['embedding_size'] = 0
    return params


def trial_command(args, params, trial_dir):
    command = [
        args.python, 'main.py',
        '--rank', '1', '--model_name', MODEL,
        '--optimizer', 'Adagrad', '--lr', str(params['lr']),
        '--dataset', args.dataset, '--metric', METRICS,
        '--test_neg_n', '100', '--max_his', '5',
        '--batch_size', str(params['batch_size']),
        '--eval_batch_size', str(args.eval_batch_size),
        '--l2', '0', '--dropout', '0.0',
        '--epoch', str(args.epoch), '--early_stop', '1', '--check_epoch', '5',
        '--random_seed', str(args.training_seed), '--gpu', str(args.gpu),
        '--log_file', str(trial_dir / 'run.log'),
        '--result_file', str(trial_dir / 'result.npy'),
        '--model_path', str(trial_dir / 'model.pt'),
        '--u_vector_size', '64', '--i_vector_size', '64',
        '--hidasi_layers', params['layers'],
        '--hidasi_loss', params['loss'],
        '--hidasi_embedding', str(params['embedding_size']),
        '--hidasi_constrained_embedding',
        str(int(params['constrained_embedding'])),
        '--hidasi_dropout_embed', str(params['dropout_embed']),
        '--hidasi_dropout_hidden', str(params['dropout_hidden']),
        '--hidasi_momentum', str(params['momentum']),
        '--hidasi_bpreg', str(params['bpreg']),
        '--hidasi_elu_param', str(params['elu_param']),
        '--hidasi_adagrad_eps', str(params['adagrad_eps']),
    ]
    return command


MODEL = 'GRU4RecHidasi'
