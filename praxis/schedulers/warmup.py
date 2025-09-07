"""Linear warmup scheduler."""

import torch


class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear warmup followed by constant learning rate.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        last_epoch: The index of the last epoch
    """
    
    def __init__(self, optimizer, warmup_steps=1024, last_epoch=-1):
        def lr_lambda_with_warmup(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        super().__init__(optimizer, lr_lambda_with_warmup, last_epoch=last_epoch)