from typing import Union

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class WarmupCosineScheduler(_LRScheduler, AbsBatchStepScheduler):
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_epochs: Union[int, float] = None,
        warmup_steps: Union[int, float] = None,
        num_epochs: Union[int, float] = 50,
        last_epoch:int = -1
    ):
        assert check_argument_types()
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        
        self.warmup_iter = 99999999 # Temporary
        self.decay_iter = 99999999
        
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def set_iter_per_epoch(self, iter_per_epoch:int):
        """This method MUST be called before using the scheduler."""
        self.iter_per_epoch = iter_per_epoch
        self.warmup_iter = self.warmup_epochs * iter_per_epoch if self.warmup_epochs is not None else self.warmup_steps
        self.total_iter = self.num_epochs * iter_per_epoch
        self.decay_iter = self.total_iter - self.warmup_iter
        
    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_epochs={self.warmup_epochs})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        
        lrs = [
            lr
            * min(step_num / self.warmup_iter if self.warmup_iter else 1, 
                    0.5 * (1 + np.cos(np.pi * (step_num - self.warmup_iter) / self.decay_iter)))
            for lr in self.base_lrs
        ]
        
        return lrs