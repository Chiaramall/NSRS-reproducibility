"""Shared training commands and log parsing."""
import re

METRICS = 'ndcg@5,ndcg@10,hit@5,hit@10'


BEST_RE = re.compile(
    r'Best Iter\(validation\)=\s*(\d+).*?'
    r'valid=\s*([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)\s+'
    r'test=\s*([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)'
)


def append_model_args(command, model, params):
    if model in ('BPR-MF', 'SVD++'):
        size = params['embedding_size']
        command.extend(['--u_vector_size', str(size), '--i_vector_size', str(size)])
        if model == 'SVD++':
            command.extend(['--max_his', '-1', '--sparse_his', '0'])
        return

    if model == 'DMF':
        projection = params['projection_size']
        command.extend([
            '--u_vector_size', '64', '--i_vector_size', '64',
            '--dmf_user_embedding_size', str(projection),
            '--dmf_item_embedding_size', str(projection),
            '--dmf_user_layers', params['tower_layers'],
            '--dmf_item_layers', params['tower_layers'],
            '--dmf_inter_matrix_type', '01',
        ])
        return

    if model == 'NeuMF':
        command.extend([
            '--u_vector_size', '64', '--i_vector_size', '64',
            '--neumf_mf_vector_size', str(params['mf_embedding_size']),
            '--neumf_mlp_vector_size', str(params['mlp_embedding_size']),
            '--neumf_mlp_layers', params['mlp_layers'],
        ])
        return

    if model == 'STAMP':
        size = params['embedding_size']
        mode = params['negative_mode']
        neg_emb = 1 if mode == 'separate_embedding' else 0
        neg_layer = '[%d]' % size if mode == 'mlp' else '[]'
        command.extend([
            '--u_vector_size', str(size), '--i_vector_size', str(size),
            '--attention_size', str(params['attention_size']),
            '--max_his', str(params['max_his']),
            '--hidden_size', '64', '--num_layers', '1', '--p_layers', '[64]',
            '--neg_emb', str(neg_emb), '--neg_layer', neg_layer,
        ])
        return

    raise ValueError('Unsupported model: %s' % model)


def trial_command(args, model, params, trial_dir):
    dropout = params.get('dropout', 0.2)
    command = [
        args.python, 'main.py',
        '--rank', '1', '--model_name', model,
        '--optimizer', 'Adam', '--lr', str(params['lr']),
        '--dataset', args.dataset, '--metric', METRICS,
        '--test_neg_n', '100',
        '--batch_size', str(params['batch_size']),
        '--eval_batch_size', str(args.eval_batch_size),
        '--l2', str(params['l2']), '--dropout', str(dropout),
        '--epoch', str(args.epoch), '--early_stop', '1', '--check_epoch', '5',
        '--random_seed', str(args.training_seed), '--gpu', str(args.gpu),
        '--log_file', str(trial_dir / 'run.log'),
        '--result_file', str(trial_dir / 'result.npy'),
        '--model_path', str(trial_dir / 'model.pt'),
    ]
    append_model_args(command, model, params)
    return command


def read_best_metrics(log_file):
    text = log_file.read_text(errors='ignore')
    matches = BEST_RE.findall(text)
    if not matches:
        raise RuntimeError('Best validation metrics not found in %s' % log_file)
    values = matches[-1]
    return {
        'epoch': int(values[0]),
        'validation': [float(value) for value in values[1:5]],
        'test': [float(value) for value in values[5:9]],
    }

