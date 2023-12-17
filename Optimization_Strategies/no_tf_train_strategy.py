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
            self.get_err_r = self.kratos_simulation.get_err_rnorm
        elif strategy_config["r_loss_type"]=='diff':
            self.get_err_r = self.kratos_simulation.get_err_rdiff

        if strategy_config["r_loss_log_scale"]==True:
            self.r_loss_scale = self.log_loss_scale
        else:
            self.r_loss_scale = self.default_loss_scale

        self.run_eagerly = False

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")
    
    @tf.function
    def default_loss_scale(self, err_r):
        loss_r = tf.linalg.matmul(err_r,err_r,transpose_b=True)
        return loss_r
    
    @tf.function
    def log_loss_scale(self, err_r):
        err_r_quad = tf.linalg.matmul(err_r,err_r,transpose_b=True)
        loss_r = tf.math.log(err_r_quad+1)
        return loss_r
    
    def generate_gradient_sum_functions(self):
        return

    def train_step(self,data):
        ## We already obtained the trained model via de PrePostProcessor, so there is nothing to train
        ## We just get the loss for one epoch and print it

        # input_batch, (target_snapshot_batch,target_aux_batch,snapshot_bound_batch) = data # target_aux is the reference force or residual, depending on the settings
        input_batch, (target_snapshot_batch,target_aux_batch) = data # target_aux is the reference force or residual, depending on the settings

        batch_len=input_batch.shape[0]

        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):
            
            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)
            # snapshot_bound=tf.expand_dims(snapshot_bound_batch[sample_id],axis=0)

            x_pred = self(input, training=False)
            # x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,snapshot_bound))
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,None))

            err_x = target_snapshot - x_pred_denorm
            loss_x = tf.linalg.matmul(err_x,err_x,transpose_b=True)

            err_r = self.get_err_r(x_pred_denorm,target_aux)
            loss_r = self.r_loss_scale(err_r)

            total_loss_x+=loss_x/batch_len
            total_loss_r+=loss_r/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}

    def test_step(self, data):
        # input_batch, (target_snapshot_batch,target_aux_batch,snapshot_bound_batch) = data
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        batch_len=input_batch.shape[0]

        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):

            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)
            # snapshot_bound=tf.expand_dims(snapshot_bound_batch[sample_id],axis=0)

            x_pred = self(input, training=False)
            # x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,snapshot_bound))
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,None))

            err_x = target_snapshot - x_pred_denorm
            loss_x = tf.linalg.matmul(err_x,err_x,transpose_b=True)

            err_r = self.get_err_r(x_pred_denorm, target_aux)
            loss_r = self.r_loss_scale(err_r)

            total_loss_x+=loss_x/batch_len
            total_loss_r+=loss_r/batch_len

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
