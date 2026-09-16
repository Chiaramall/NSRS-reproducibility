"""Model-specific search space and training arguments."""
from hyperopt import hp
from common import METRICS, trial_command

def common_space(prefix):
    return {
        'lr': hp.choice(prefix + '_lr', [1e-4, 5e-4, 1e-3, 5e-3]),
        'l2': hp.choice(prefix + '_l2', [1e-6, 1e-5, 1e-4, 1e-3]),
        'dropout': hp.choice(prefix + '_dropout', [0.0, 0.1, 0.2, 0.3, 0.5]),
        'batch_size': hp.choice(prefix + '_batch_size', [64, 128, 256, 512]),
        'hidden_size': hp.choice(prefix + '_hidden_size', [32, 64, 128, 256]),
    }


def search_space(model):
    prefix = model.lower()
    space = common_space(prefix)
    if model == 'SASRec':
        space['num_blocks'] = hp.choice(prefix + '_num_blocks', [1, 2, 3])
        space['num_heads'] = hp.choice(prefix + '_num_heads', [1, 2, 4])
        space['norm_first'] = hp.choice(prefix + '_norm_first', [0, 1])
    elif model == 'BERT4Rec':
        space['num_blocks'] = hp.choice(prefix + '_num_blocks', [1, 2, 3])
        space['num_heads'] = hp.choice(prefix + '_num_heads', [1, 2, 4])
        space['intermediate_size'] = hp.choice(
            prefix + '_intermediate_size', [64, 128, 256, 512])
        space['pooling'] = hp.choice(prefix + '_pooling', ['last', 'mean'])
    else:
        raise ValueError('Unsupported model: %s' % model)
    return space


def normalized_params(params):
    params = dict(params)
    hidden = int(params['hidden_size'])
    heads = int(params['num_heads'])
    if hidden % heads != 0:
        # Keep trials valid while preserving the sampled hidden size.
        valid_heads = [head for head in [4, 2, 1] if hidden % head == 0]
        params['num_heads'] = valid_heads[0]
    return params


def trial_command(args, model, params, trial_dir):
    command = [
        args.python, 'main.py',
        '--rank', '1', '--model_name', model,
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
        '--u_vector_size', str(params['hidden_size']),
        '--i_vector_size', str(params['hidden_size']),
    ]
    if model == 'SASRec':
        command.extend([
            '--sasrec_num_blocks', str(params['num_blocks']),
            '--sasrec_num_heads', str(params['num_heads']),
            '--sasrec_norm_first', str(params['norm_first']),
        ])
    elif model == 'BERT4Rec':
        command.extend([
            '--bert_num_blocks', str(params['num_blocks']),
            '--bert_num_heads', str(params['num_heads']),
            '--bert_intermediate_size', str(params['intermediate_size']),
            '--bert_pooling', params['pooling'],
        ])
    else:
        raise ValueError('Unsupported model: %s' % model)
    return command

