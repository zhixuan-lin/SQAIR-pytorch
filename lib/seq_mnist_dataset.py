import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class SequentialMNIST(Dataset):
    """
    Sequential multi-mnist dataset
    """
    def __init__(self, root, mode, seq_len):
        """
        Args:
            root: directory path
            mode: either ['train', 'valid']
            len: sequence length
        """
        assert mode in ['train', 'valid'], 'Invalid dataset mode'
        Dataset.__init__(self)
        path = {
            'train': 'seq_mnist_train.pickle',
            'valid': 'seq_mnist_validation.pickle'
        }[mode]
        # Path to the pickle file
        path = os.path.join(root, path)
        # Load dataset
        with open(path, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
            
        # (T, N, H, W)
        self.imgs = dataset['imgs'][:seq_len]
        # (N, 2), numbers in the digits
        self.labels = dataset['labels']
        # (T, N, 2, 4), the last dimension being [x, y, w, h]
        self.coords = dataset['coords']
        # (1, N, 3), bool
        self.nums = dataset['nums']
        # (1, N, 3) -> (1, N)
        self.nums = np.sum(self.nums, axis=-1)
        
    def __getitem__(self, index):
        """
        Args:
            index: integer
        Returns:
            A tuple (imgs, nums).
            
            imgs: (T, 1, C, H, W), FloatTensor in range (0, 1)
            nums: (T, 1), int, number of digits for each time step
        """
        
        # (T, H, W)
        imgs = self.imgs[:, index]
        # (1,)
        nums = self.nums[:, index].astype(np.float)
        
        imgs = imgs.astype(np.float) / 255.0
        imgs = torch.from_numpy(imgs).float()
        nums = torch.from_numpy(nums).float()
        
        T = imgs.size(0)
        # (T, 1, C, H, W)
        imgs = imgs[:, None, None]
        # (1, 1)
        # (1, 1) -> (T, 1)
        nums = nums.expand(T, 1)
        
        
        return imgs, nums
    
    def __len__(self):
        return self.imgs.shape[1]
    
    
def collate_fn(samples):
    """
    collate_fn for SequentialMNIST.
    
    Args:
        samples: a list of samples. Each item is a (imgs, nums) pair. Where
        
            - imgs: shape (T, 1, C, H, W)
            - nums: shape (T, 1)
            
            And len(samples) is the batch size.

    Returns:
        A tuple (imgs, nums). Where
        
        - imgs: shape (T, B, C, H, W)
        - nums: shape (T, B)
    """
    imgs, nums = zip(*samples)
    # (T, B, C, H, W)
    imgs = torch.cat(imgs, dim=1)
    # (T, B)
    nums = torch.cat(nums, dim=1)
    
    return imgs, nums
    
        
        
