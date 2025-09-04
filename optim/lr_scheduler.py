import math
import warnings
from typing import Optional, Union

from torch import inf
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from utils import verbose


class EmptyScheduler:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def step(self):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        if state_dict is not None:
            raise KeyError("Tried to load a non-empty scheduler to an empty scheduler")


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 1e-4.
        initial_lr(float): Initial Learning rate. Default: min_lr
        warmup_steps(int): Linear warmup step size. Default: 0.
        warmup_lr_init(float): Initial lr. Used if warmup_steps > 0. Default: 1e-4
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.0001,
                 initial_lr: Optional[float] = None,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.initial_lr = initial_lr if initial_lr is not None else min_lr
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # current cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.base_lrs = []
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
            self.base_lrs.append(self.initial_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        initial_patience (int): Number of epochs to maintain initial
            learning rate.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 initial_patience=0,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.initial_patience = initial_patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if epoch <= self.initial_patience:
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            return

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class CosineAnnealingWarmup(_LRScheduler):
    """ CosineAnnealing with Warmup.
    self.warmup_step() should be called at the end of each iteration.
    self.step() should be called at the end of each epoch.
    Example:
    >>> for epoch in range(max_epochs):
    >>>     for iteration, batch in enumerate(dataloader):
    >>>         train_iteration(batch)
    >>>         optim.step()
    >>>         scheduler.warmup_step()
    >>>     scheduler.step()
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iterations: int,
        T_max: Union[int, float],
        eta_min: float = 0.0,
        warn: bool = True
    ):
        self.current_iteration = 1
        self.warmup_iterations = warmup_iterations
        self.T_max = T_max
        self.eta_min = eta_min
        self.warn = warn
        super().__init__(optimizer)

    def warmup_step(self) -> None:
        """ should be at the enf of the training loop,
        after the first optim.step() """
        if self.current_iteration > self.warmup_iterations:
            return
        
        scale = self.current_iteration / self.warmup_iterations
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * scale

        self.current_iteration += 1
    
    def get_lr(self):
        if self.current_iteration <= self.warmup_iterations:
            scale = self.current_iteration / self.warmup_iterations
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr in self.base_lrs]

    def step(self):
        if self.last_epoch == -1 or self.current_iteration > self.warmup_iterations:
            super().step()
            return
        
        # warmup_iterations is longer than an epoch.
        # Don't increase last_epoch and decrease self.T_max by 1.
        if self.warn and verbose():
            warnings.warn(
                f"CosineAnnealingWarmup.warmup_iterations is longer than the "
                f"iterations of an epoch. T_max is decreased by 1."
            )
        self.T_max -= 1
        return
