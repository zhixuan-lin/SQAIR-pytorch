import torch
import sys
import argparse
from yacs.config import CfgNode
from lib.seq_mnist_dataset import SequentialMNIST



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/defaults.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        'opts',
        help='Modify options using command line',
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    # # Load configuration file
    cfg = CfgNode(new_allowed=True)
    with open('configs/defaults.yaml', 'r') as f:
        cfg.load_cfg(f)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    dataset = SequentialMNIST(root=cfg.dataset.seq_mnist, mode='train')

