import torch
import sys
import argparse
from yacs.config import CfgNode
from lib.seq_mnist_dataset import SequentialMNIST, collate_fn
from torch.utils.data import DataLoader
from lib.vimco import SQAIRVIMCO



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
    # cfg = CfgNode(new_allowed=True)
    # with open('configs/defaults.yaml', 'r') as f:
    #     cfg.load_cfg(f)
    # cfg.merge_from_file(args.config)
    # cfg.merge_from_list(args.opts)
    
    trainset = SequentialMNIST(root=cfg.dataset.seq_mnist, mode='valid')
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4,
                             collate_fn=collate_fn)


    model = SQAIRVIMCO()
    model.train()
    
    for i, data in enumerate(trainloader):
        imgs, nums = data
        loss = model(imgs)
        print(i, loss.item())
    
    

