from attrdict import AttrDict
import os

cfg = AttrDict({
    # 'exp_name': 'test',
    # 'exp_name': 'anneal',
    # 'exp_name': 'weight',
    # 'exp_name': 'double_anneal',
    'exp_name': 'double_anneal_from05',
    # 'exp_name': 'no_rein',
    # 'exp_name': 'test_05',
    # 'exp_name': 'noreinforce',
    'resume': True,
    'device': 'cpu',
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
    'multi_mnist_path': os.path.join('data', 'multi_mnist')
})
