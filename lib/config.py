from attrdict import AttrDict
import os

cfg = AttrDict({
    # 'exp_name': 'test-len10-delta',
    # 'exp_name': 'test-len1-fixedscale-aggre-super',
    # 'exp_name': 'test-aggre-super',
    # 'exp_name': 'test-mask',
    'exp_name': 'test-proposal',
    'resume': True,
    'device': 'cuda:3',
    # 'device': 'cpu',
    'dataset': {
        'seq_mnist': 'data/seq_mnist',
        'seq_len': 2
    },
    
    'train': {
        'batch_size': 32,
        'model_lr': 1e-4,
        'max_epochs': 1000
    },
    'valid': {
        'batch_size': 64
    },
    'anneal': {
        'initial': 0.70,
        'final': 0.01,
        'total_steps':40000,
        'interval': 500
    },
    'logdir': 'logs/',
    'checkpointdir': 'checkpoints/',
})
