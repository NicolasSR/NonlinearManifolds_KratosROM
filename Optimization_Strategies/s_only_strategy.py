import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class S_Only_Strategy_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(S_Only_Strategy_KerasModel,self).__init__(*args,**kwargs)

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

        self.sample_gradient_sum_functions_list=None
    
    @tf.function
    def default_loss_scale(self, err_r):
        loss_r = tf.linalg.matmul(err_r,err_r,transpose_b=True)
        return loss_r
    
    @tf.function
    def log_loss_scale(self, err_r):
        err_r_quad = tf.linalg.matmul(err_r,err_r,transpose_b=True)
        loss_r = tf.math.log(err_r_quad+1)
        return loss_r
    
    @tf.function
    def get_v_loss_x(self, input, target_snapshot):
        x_pred = self(input, training=True)
        x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, input)
        v_loss_x = x_pred_denorm - target_snapshot  # We get the loss on the error for the denormalised snapshot
        return v_loss_x, x_pred_denorm
    
    @tf.function
    def get_gradients(self, trainable_vars, input, v_loss_x):

        v_loss = 2*v_loss_x

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(input, training=True)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, input)
            v_u_dotprod = tf.linalg.matmul(v_loss, x_pred_denorm, transpose_b=True)
        grad_loss=tape_d.gradient(v_u_dotprod, trainable_vars)

        return grad_loss

    def generate_gradient_sum_functions(self):
        @tf.function
        def gradient_sum_sample(previous_gradients, gradients, batch_len):
            updated_gradients=previous_gradients+gradients/batch_len
            return updated_gradients
        
        self.sample_gradient_sum_functions_list=[]
        for i in range(len(self.trainable_variables)):
            self.sample_gradient_sum_functions_list.append(gradient_sum_sample)

    def train_step(self,data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data # target_aux is the reference force or residual, depending on the settings
        trainable_vars = self.trainable_variables

        batch_len=input_batch.shape[0]

        total_gradients=[]
        for i in range(len(trainable_vars)):
            total_gradients.append(tf.zeros_like(trainable_vars[i]))
        
        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):
            
            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

            v_loss_x, x_pred_denorm = self.get_v_loss_x(input, target_snapshot)
            loss_x = tf.linalg.matmul(v_loss_x,v_loss_x,transpose_b=True)

            err_r = self.get_err_r(x_pred_denorm,target_aux)
            loss_r = self.r_loss_scale(err_r)

            total_loss_x+=loss_x/batch_len
            total_loss_r+=loss_r/batch_len

            grad_loss = self.get_gradients(trainable_vars, input, v_loss_x)

            for i in range(len(total_gradients)):
                total_gradients[i]=self.sample_gradient_sum_functions_list[i](total_gradients[i], grad_loss[i], batch_len)

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}

    def test_step(self, data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        batch_len=input_batch.shape[0]

        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):

            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

            x_pred = self(input, training=False)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred,input)

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
