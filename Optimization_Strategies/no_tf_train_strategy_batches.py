import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class NoTFTrain_Strategy_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(NoTFTrain_Strategy_KerasModel,self).__init__(*args,**kwargs)

        self.prepost_processor = prepost_processor
        self.kratos_simulation = kratos_simulation

        if strategy_config["r_loss_type"]=='norm':
            self.get_err_r = self.kratos_simulation.get_err_rnorm_batch
        elif strategy_config["r_loss_type"]=='diff':
            self.get_err_r = self.kratos_simulation.get_err_rdiff_batch

        if strategy_config["r_loss_log_scale"]==True:
            self.r_loss_scale = self.log_loss_scale
        else:
            self.r_loss_scale = self.default_loss_scale

        self.run_eagerly = False

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")
    
    @tf.function
    def default_loss_scale(self, err_r):
        loss_r = tf.math.reduce_sum(tf.math.square(err_r), axis=1)
        return loss_r
    
    @tf.function
    def log_loss_scale(self, err_r):
        err_r_quad = tf.math.reduce_sum(tf.math.square(err_r), axis=1)
        loss_r = tf.math.log(err_r_quad+1)
        return loss_r
    
    def generate_gradient_sum_functions(self):
        return
    
    def update_rescaling_factors(self, S_true, R_true):
        pass

    def train_step(self,data):
        ## We already obtained the trained model via de PrePostProcessor, so there is nothing to train
        ## We just get the loss for one epoch and print it

        return self.test_step(data)

    def test_step(self, data):
        # input_batch, (target_snapshot_batch,target_aux_batch,snapshot_bound_batch) = data
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        x_pred_batch = self(input_batch, training=False)
        x_pred_denorm_batch = self.prepost_processor.postprocess_output_data_tf(x_pred_batch,(input_batch,None))
        err_x_batch = target_snapshot_batch - x_pred_denorm_batch
        loss_x_batch = tf.math.reduce_sum(tf.math.square(err_x_batch), axis=1)

        err_r_batch = self.get_err_r(x_pred_denorm_batch, target_aux_batch)
        loss_r_batch = self.r_loss_scale(err_r_batch)

        total_loss_x=tf.math.reduce_mean(loss_x_batch)
        total_loss_r=tf.math.reduce_mean(loss_r_batch)

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        
        return [self.loss_x_tracker, self.loss_r_tracker]
