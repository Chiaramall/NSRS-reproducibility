"""Model-specific search space and training arguments."""
from hyperopt import hp
from common import METRICS, trial_command

def common_space(prefix):
    return {
        'lr': hp.choice(prefix + '_lr', [1e-5, 1e-4, 1e-3, 1e-2]),
        'l2': hp.choice(prefix + '_l2', [
            1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
        ]),
        'batch_size': hp.choice(
            prefix + '_batch_size', [32, 64, 128, 256, 512, 1024]),
    }


def search_space(model):
    prefix = model.replace('+', 'plus').replace('-', '_')
    space = common_space(prefix)

    if model == 'BPR-MF':
        space['embedding_size'] = hp.choice(
            prefix + '_embedding_size', [16, 32, 64, 128, 256, 512])
    elif model == 'SVD++':
        space['embedding_size'] = hp.choice(
            prefix + '_embedding_size', [16, 32, 64, 128, 256])
    elif model == 'DMF':
        space['dropout'] = hp.choice(
            prefix + '_dropout', [0.0, 0.1, 0.2, 0.3, 0.5, 0.6])
        space['projection_size'] = hp.choice(
            prefix + '_projection_size', [32, 64, 128, 256, 512])
        space['tower_layers'] = hp.choice(prefix + '_tower_layers', [
            '[32]', '[64]', '[128]', '[256]', '[512]',
            '[256,128]', '[512,256]', '[256,128,64]',
        ])
    elif model == 'NeuMF':
        space['dropout'] = hp.choice(
            prefix + '_dropout', [0.0, 0.1, 0.2, 0.3, 0.5, 0.6])
        space['mf_embedding_size'] = hp.choice(
            prefix + '_mf_embedding_size', [8, 16, 32, 64, 128, 256])
        space['mlp_embedding_size'] = hp.choice(
            prefix + '_mlp_embedding_size', [4, 8, 16, 32, 64, 128])
        space['mlp_layers'] = hp.choice(prefix + '_mlp_layers', [
            '[16,8,4]', '[32,16,8]', '[64,32,16]',
            '[128,64,32]', '[256,128,64]', '[128,64,32,16]',
        ])
    elif model == 'STAMP':
        space['embedding_size'] = hp.choice(
            prefix + '_embedding_size', [16, 32, 64, 128, 256])
        space['attention_size'] = hp.choice(
            prefix + '_attention_size', [16, 32, 64, 128, 256, 512])
        space['max_his'] = hp.choice(
            prefix + '_max_his', [3, 5, 10, 20, 50])
        space['negative_mode'] = hp.choice(prefix + '_negative_mode', [
            'separate_embedding', 'sign_inversion', 'mlp',
        ])
    else:
        raise ValueError('Unsupported model: %s' % model)
    return space

