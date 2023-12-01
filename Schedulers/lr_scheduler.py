
import numpy as np
import tensorflow as tf

import keras
from keras import backend
from keras.utils import io_utils

def get_lr_schedule_func(lr_config):

    def lr_sgdr_schedule(epoch, lr):
        #('sgdr', 0.001, 1e-6, 250, 2)# References
        """
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
        """
        if epoch==0:
            new_lr=lr_config[1]
        else:
            max_lr=lr_config[1]
            min_lr=lr_config[2]
            cycle_length=lr_config[3]
            scale_factor=lr_config[4]
            # cycle=np.floor(1+epoch/cycle_length)
            cycle=np.floor(epoch/cycle_length)
            x=np.abs(epoch/cycle_length-cycle)
            # new_lr = min_lr+(max_lr-min_lr)*0.5*(1+np.cos(x*np.pi))/scale_factor**(cycle-1)
            new_lr = min_lr+(max_lr-min_lr)*0.5*(1+np.cos(x*np.pi))/scale_factor**cycle
        return new_lr

    def lr_tri2_schedule(epoch, lr):
        #('tri2', 0.001, 1e-6, 250, 2)
        if epoch==0:
            new_lr=lr_config[2]
        else:
            max_lr=lr_config[1]
            init_lr=lr_config[2]
            step_size=lr_config[3]
            scale_factor=lr_config[4]
            cycle=np.floor(1+epoch/(2*step_size))
            x=np.abs(epoch/step_size-2*cycle+1)
            new_lr=init_lr+(max_lr-init_lr)*np.max([0,1-x])/scale_factor**(cycle-1)
        return new_lr

    def lr_steps_schedule(epoch, lr):
        if epoch==0:
            new_lr=lr_config[1]
        elif epoch%lr_config[4]==0:
            new_lr=max(lr/lr_config[2], lr_config[3])
        else:
            new_lr=lr
        return new_lr

    def lr_const_schedule(epoch, lr):
        new_lr=lr_config[1]
        return new_lr

    if lr_config[0]=='const':
        lr_schedule_func = lr_const_schedule
    elif lr_config[0]=='steps':
        lr_schedule_func = lr_steps_schedule
    elif lr_config[0]=='tri2':
        lr_schedule_func = lr_tri2_schedule
    elif lr_config[0]=='sgdr':
        lr_schedule_func = lr_sgdr_schedule
    else:
        print('Unvalid lr scheduler')
        lr_schedule_func = None

    return lr_schedule_func

class LR_Scheduler(keras.callbacks.Callback):
    """ Custom Learning rate scheduler.
    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.
    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
          and current learning rate (float) as inputs and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages. """

    def __init__(self, lr_schedule, verbose=0):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.lr_schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LR_Scheduler setting"
                f"learning rate to {lr}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)