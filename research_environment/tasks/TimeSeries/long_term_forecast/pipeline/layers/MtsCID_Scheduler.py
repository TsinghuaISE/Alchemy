"""
MtsCID Learning Rate Scheduler

This module contains learning rate schedulers extracted from the MtsCID model.
These schedulers are model-agnostic and can be reused by other models.
"""

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    """
    Polynomial learning rate decay scheduler with warmup.
    
    The learning rate starts from 0 and linearly increases to `lr` during the warmup phase,
    then decays polynomially from `lr` to `end_lr` over the remaining training steps.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_updates: Number of warmup steps
        tot_updates: Total number of training steps
        lr: Peak learning rate (after warmup)
        end_lr: Final learning rate
        power: Polynomial decay power (default: 1.0 for linear decay)
        last_epoch: The index of last epoch (default: -1)
        verbose: If True, prints a message to stdout for each update (default: False)
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = PolynomialDecayLR(
        ...     optimizer,
        ...     warmup_updates=1000,
        ...     tot_updates=10000,
        ...     lr=0.002,
        ...     end_lr=0.00005,
        ...     power=1.0
        ... )
        >>> for epoch in range(num_epochs):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_updates,
        tot_updates,
        lr,
        end_lr,
        power=1.0,
        last_epoch=-1,
        verbose=False
    ):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Compute learning rate using polynomial decay schedule with warmup.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self._step_count <= self.warmup_updates:
            # Warmup phase: linear increase from 0 to lr
            warmup_factor = self._step_count / float(self.warmup_updates) if self.warmup_updates > 0 else 1.0
            lr = warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            # After total updates: use end_lr
            lr = self.end_lr
        else:
            # Decay phase: polynomial decay from lr to end_lr
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot_updates - warmup)
            lr = lr_range * (pct_remaining ** self.power) + self.end_lr
        
        return [lr for _ in self.optimizer.param_groups]

