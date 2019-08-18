from attrdict import AttrDict
import os

cfg = AttrDict({
    'exp_name': 'test-len2',
    'resume': True,
    'device': 'cuda:3',
    'dataset': {
        'seq_mnist': 'data/seq_mnist',
        'seq_len': 2
    },
    
    'train': {
        'batch_size': 64,
        'model_lr': 1e-4,
        'baseline_lr': 1e-1,
        'max_epochs': 1000
    },
    'valid': {
        'batch_size': 64
    },
    'anneal': {
        'initial': 0.5,
        'final': 0.01,
        'total_steps':40000,
        'interval': 500
    },
    'logdir': 'logs/',
    'checkpointdir': 'checkpoint/',
})
