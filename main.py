import torch
torch.manual_seed(233)
import sys
import os
import argparse
from yacs.config import CfgNode
from lib.seq_mnist_dataset import SequentialMNIST, collate_fn
from torch.utils.data import DataLoader
from lib.vimco import SQAIRVIMCO
from lib.config import cfg
from torch.optim import Adam
from tensorboardX import SummaryWriter
from lib.utils import vis_logger, metric_logger, WeightScheduler, Checkpointer



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
    
    trainset = SequentialMNIST(root=cfg.dataset.seq_mnist, mode='train', seq_len=cfg.dataset.seq_len)
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4,
                             collate_fn=collate_fn)


    model = SQAIRVIMCO().to(cfg.device)
    model.train()
    
    optimizer = Adam(model.parameters(), lr=cfg.train.model_lr)
    
    
    weight_scheduler = WeightScheduler(0.0, 1.0, 10000, 20000, 500, model, cfg.device)
    
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name))

    start_epoch = 0
    if cfg.resume:
        start_epoch = checkpointer.load(model, optimizer)
        
    writer = SummaryWriter(logdir=os.path.join(cfg.logdir, cfg.exp_name), flush_secs=30)
    
    print('Start training')
    for epoch in range(start_epoch, cfg.train.max_epochs):
        for i, data in enumerate(trainloader):
            global_step = epoch * len(trainloader) + i + 1
            imgs, nums = data
            imgs = imgs.to(cfg.device)
            nums = nums.to(cfg.device)
            loss = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weight_scheduler.step()
            
            metric_logger.update(loss=loss.item())

            if (i + 1) % 50 == 0:
                print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.2f}'.format(
                    epoch + 1, cfg.train.max_epochs, i + 1, len(trainloader), metric_logger['loss'].median))
                vis_logger.add_to_tensorboard(writer, global_step)
                writer.add_scalar('loss/loss', metric_logger['loss'].median, global_step)
                writer.add_scalar('other/reinforce_weight', model.reinforce_weight, global_step)
                # writer.add_scalar('accuracy/train_total', acc_total, global_step)
                # writer.add_scalar('accuracy/train_zero', acc_zero, global_step)
                # writer.add_scalar('accuracy/train_one', acc_one, global_step)
                # writer.add_scalar('accuracy/train_two', acc_two, global_step)
                # writer.add_scalar('other/pres_prior_prob', prior_scheduler.current, global_step)
                # writer.add_scalar('other/reinforce_weight', weight_scheduler.current, global_step)
                

        if epoch % 2 == 0:
            checkpointer.save(model, optimizer, epoch+1)


            

