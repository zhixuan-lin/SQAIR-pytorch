from collections import defaultdict, deque
import pickle
from attrdict import AttrDict
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
from matplotlib import patches
from matplotlib import pyplot as plt
from torch.distributions.bernoulli import Bernoulli


# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10

class VisLogger:
    """
    Global visualization logger
    """
    def __init__(self):
        self.things = {}
        
    def update(self, **kargs):
        for key, value in kargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.things.update(value)
            
    def __getitem__(self, key):
        return self.things[key]
    
    def __setitem__(self, key, item):
        self.things[key] = item
        
    def add_to_tensorboard(self, writer: SummaryWriter, global_step):
        """
        Process data and return a dictionary
        vis_logger['canvas'].append(vis_logger['canvas_cur'])
        vis_logger['z_pres'].append(vis_logger['z_pres_cur'])
        vis_logger['z_pres_prob'].append(vis_logger['z_pres_prob_cur'])
        vis_logger['z_where'].append(vis_logger['z_where_cur'])
        vis_logger['id'].append(vis_logger['id_cur'])
        'imgs'
        
        kl_z_pres_list: list of scalars
        kl_z_where_list:
        kl_z_what_list
        kl: scalar
        likelihood: scalar
        reinforce_term: scalar
        elbo: scalar
        """
        
        # losses
        kl_pres = torch.sum(torch.tensor(self.things['kl_pres_list'])).item()
        kl_where = torch.sum(torch.tensor(self.things['kl_where_list'])).item()
        kl_what = torch.sum(torch.tensor(self.things['kl_what_list'])).item()
        #
        kl_total = self.things['kl']
        # baseline_loss = self.things['baseline_loss']
        neg_reinforce = -self.things['reinforce_term']
        neg_likelihood = -self.things['likelihood']
        neg_elbo = -self.things['elbo']
        #
        writer.add_scalar('kl/kl_pres', kl_pres, global_step)
        writer.add_scalar('kl/kl_where', kl_where, global_step)
        writer.add_scalar('kl/kl_what', kl_what, global_step)
        writer.add_scalar('loss/kl_total', kl_total, global_step)
        # writer.add_scalar('loss/baseline_loss', baseline_loss, global_step)
        writer.add_scalar('loss/neg_reinforce', neg_reinforce, global_step)
        writer.add_scalar('loss/neg_likelihood', neg_likelihood, global_step)
        writer.add_scalar('loss/neg_elbo', neg_elbo, global_step)
        
        imgs = [x.detach().cpu().numpy() for x in self.things['imgs']]
        canvas = [[x.detach().cpu().numpy() for x in y] for y in self.things['canvas']]
        z_pres = [[x.detach().cpu().item() for x in y] for y in self.things['z_pres']]
        z_pres_prob = [[x.detach().cpu().item() for x in y] for y in self.things['z_pres_prob']]
        id = [[x.detach().cpu().item() for x in y] for y in self.things['id']]
        z_where = [[x.detach().cpu().numpy() for x in y] for y in self.things['z_where']]
        
        # image = self.things['image']
        # writer.add_image('vis/original', image.detach(), global_step)
        fig = create_fig(imgs, canvas, z_pres, z_pres_prob, z_where, id)
        fig.show()
        # fig.show()
        # writer.add_scalar('train', global_step, global_step)
        writer.add_figure('vis/reconstruct', fig, global_step)
        plt.close(fig)
        
def create_fig(imgs, canvas, z_pres, z_prob, z_where, id):
    """
    Args:
        imgs: a list of T images. Each being (H, W)
        canvas: a list of list. List shape (T, K), each (H, W)
        z_pres: list of list of scalars
        z_prob: list of list of scalars
        z_where: list of list of (4,)
        id: list of list of scalars

    Returns:
        plt.figure

    """
    T = len(imgs)
    K = len(canvas[0])

    size = 2
    fig = plt.figure(figsize=(2 * K * size, T * size))
    # There will be (T, 2K) subplots
    for t in range(T):
    
        # Original
        for i in range(3):
            ax = fig.add_subplot(T, 2*K, 2*K * t + i + 1)
            ax.set_axis_off()
            
            ax.imshow(imgs[t], cmap='gray', vmin=0.0, vmax=1.0)
            if z_pres[t][i] == 1:
                draw_bounding_box(ax, z_where[t][i], (50, 50), colors[int(id[t][i])])
        
        # Canvas
        for i in range(3):
            ax = fig.add_subplot(T, 2*K, 2*K * t + i + 4)
            ax.set_title('pres: {:.0f}, p: {:.2f}'.format(z_pres[t][i], z_prob[t][i] if i == 0 or z_pres[t][i-1] == 1 else 0))
            ax.set_axis_off()
            ax.imshow(canvas[t][i], cmap='gray', vmin=0.0, vmax=1.0)
            if z_pres[t][i] == 1:
                draw_bounding_box(ax, z_where[t][i], (50, 50), colors[int(id[t][i])])
        
    return fig
    

def draw_bounding_box(ax: plt.Axes, z_where, size, color):
    """
    :param ax: matplotlib ax
    :param z_where: [s, x, y] s < 1
    :param size: output size, (h, w)
    """
    h, w = size
    sx, sy, x, y = z_where
    
    min, max = -1, 1
    h_box, w_box = h / sy, w / sx
    x_box = (-x / sx - min) / (max - min) * w
    y_box = (-y / sy - min) / (max - min) * h
    x_box -= w_box / 2
    y_box -= h_box / 2
    
    rect = patches.Rectangle((x_box, y_box), w_box, h_box, edgecolor=color, linewidth=3.0, fill=False)
    ax.add_patch(rect)

class SmoothedValue:
    """
    Record the last several values, and return summaries
    """
    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0
    
    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value
        
    @property
    def median(self):
        return np.median(np.array(self.values))
    
    @property
    def global_avg(self):
        return self.sum / self.count
    
class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)
        
    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)
            
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, item):
        self.values[key].update(item)
        
class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)
        
    
    def save(self, model, optimizer, epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        filename = os.path.join(self.path, 'model_{:05}.pth'.format(epoch))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if os.path.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
            
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
    
    def load(self, model, optimizer):
        """
        Return starting epoch
        """
        with open(self.listfile, 'rb') as f:
            model_list = pickle.load(f)
            if len(model_list) == 0:
                print('No checkpoint found. Starting from scratch')
                return 0
            else:
                checkpoint = torch.load(model_list[-1])
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('Load checkpoint from {}.'.format(model_list[-1]))
                return checkpoint['epoch']
            
class PriorScheduler:
    def __init__(self, initial, final, total_steps, interval, model, device):
        self.initial = initial
        self.final = final
        self.total_steps = total_steps
        self.interval = interval
        self.model = model
        self.device = device
        self.current_step = 0
        self.current = initial
        
        
    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            return
        
        if self.current_step % self.interval == 0:
            ratio = self.current_step / self.total_steps
            prob = self.initial + ratio * (self.final - self.initial)
            self.model.z_pres_prob_prior = Bernoulli(torch.tensor(prob, device=self.device))
            self.current = prob
            
class WeightScheduler:
    def __init__(self, initial, final, startsfrom, total_steps, interval, model, device):
        self.initial = initial
        self.startsfrom = startsfrom
        self.final = final
        self.total_steps = total_steps
        self.interval = interval
        self.model = model
        self.device = device
        self.current_step = 0
        self.current = initial
        
        self.start = False
    
    
    def step(self):
        self.current_step += 1
        if not self.start:
            if self.current_step < self.startsfrom:
                return
            else:
                self.start = True
                self.current_step = 0
        
        if self.current_step > self.total_steps:
            return
        
        if self.current_step % self.interval == 0:
            ratio = self.current_step / self.total_steps
            weight = self.initial + ratio * (self.final - self.initial)
            self.model.reinforce_weight = weight
            self.current = weight
            
        

vis_logger = VisLogger()
metric_logger = MetricLogger()

