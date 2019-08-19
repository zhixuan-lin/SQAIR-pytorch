from attrdict import AttrDict
import os

cfg = AttrDict({
    'exp_name': 'test-len5',
    'resume': True,
    'device': 'cuda:3',
    'dataset': {
        'seq_mnist': 'data/seq_mnist',
        'seq_len': 5
    },
    
    'train': {
        'batch_size': 64,
        'model_lr': 1e-4,
        'baseline_lr': 1e-1,
        'max_epochs': 1000
    },
    'valid': {
        'batch_ size': 64
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
