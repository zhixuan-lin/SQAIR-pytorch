"""
This module implements VIMCO optimization algorithm for SQAIR.
"""
import torch
from .sqair import SQAIR
from torch import nn
import math
from torch.nn import functional as F

class SQAIRVIMCO(nn.Module):
    def __init__(self, k=5, arch_update=None):
        nn.Module.__init__(self)
        self.sqair = SQAIR(arch_update)
        # Number of samples
        self.K = k
        
    def forward(self, x):
        """
        Args:
            x: image sequence, (T, B, 1, H, W)

        Returns:
            surrogate loss that can be directly optimized with SGD.
        """
        
        T, B, C, H, W = x.size()
        # For a single example, we should perform k forward passes.
        # We do this in parallel.
        # (T, B, C, H, W) -> (T, B, 1, C, H, W)
        x = x[:, :, None]
        # (T, B, 1, C, H, W) -> (T, B, K, C, H, W)
        x = x.expand(T, B, self.K, C, H, W)
        
        # reshape to (T, B*K, C, H, W)
        x = x.view(T, B*self.K, C, H, W)
        
        # (B*K,) and (B*K,)
        log_weights, z_pres_likelihood = self.sqair(x)
        
        # Reshape to (B, K)
        log_weights, z_pres_likelihood = [x.view(B, self.K) for x in [log_weights, z_pres_likelihood]]
        
        # IWAE loss: first term
        # (B, K) -> (B,)
        iwae_term = torch.logsumexp(log_weights, dim=1) - math.log(self.K)
        
        
        # Learning signals do require grad
        log_weights = log_weights.detach()
        # Second term
        # Naive way is to multiply iwae_term with likelihood ((B,) * (B, K))
        # For VIMCO, we expand iwae_term to (B, K), and compute a control variate
        # of shape (B, K).
        control_variate = self.control_variate(log_weights)
        # (B, K)
        learning_signal = log_weights - control_variate
        # (B, K)
        reinforce_term = learning_signal * z_pres_likelihood
        # (B,)
        reinforce_term = reinforce_term.sum(1)
        
        # Remember we are doing maximization here
        loss = (iwae_term - reinforce_term).mean()
        return loss
        
    
    def control_variate(self, log_weights):
        """
        
        Args:
            log_weights: log w's in IWAE. (B, K)

        Returns:
            control_variate of shape (B, K)
        """
        
        # Recall that IWAE is torch.logsumexp(log_weights) - log(K)
        # And we are approximate this.
        
        # Estimate log h[i]
        # (B, 1)
        log_weights_sum = log_weights.sum(dim=-1, keepdim=True)
        # (B, K)
        leave_one_out_sum = log_weights_sum - log_weights
        log_weights_estimate = leave_one_out_sum.sum(dim=-1) / (self.K - 1.0)
        
        # We replace h[i] in log_weights with out estimate
        # This implementation requires some trick.
        # (B, K, 1) (will be broadcast to (B, K, K) later)
        log_weights = log_weights[:, :, None]
        
        # This log_weights_estimate - log_weights
        replacer = torch.diag_embed(log_weights_estimate - log_weights)
        
        # (B, K, K)
        estimate = log_weights + replacer
        
        # Now reduce the second dimension
        control_variate = torch.logsumexp(estimate, dim=-2) - math.log(self.K)
        
        return control_variate
        
        
        

