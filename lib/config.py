from attrdict import AttrDict
import os

cfg = AttrDict({
    'exp_name': 'test-len10-delta',
    'resume': False,
    'device': 'cuda:2',
    # 'device': 'cpu',
    'dataset': {
        'seq_mnist': 'data/seq_mnist',
        'seq_len': 10
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
