import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class S_R_Mixed_Strategy_Cropped_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(S_R_Mixed_Strategy_Cropped_KerasModel,self).__init__(*args,**kwargs)
        self.wx=0
        self.wr=0

        self.prepost_processor = prepost_processor
        self.kratos_simulation = kratos_simulation

        if strategy_config["r_loss_type"]=='norm':
            self.get_v_loss_r = self.kratos_simulation.get_v_loss_rnorm_batch
            self.get_err_r = self.kratos_simulation.get_err_rnorm_batch
        elif strategy_config["r_loss_type"]=='diff':
            self.get_v_loss_r = self.kratos_simulation.get_v_loss_rdiff_batch
            self.get_err_r = self.kratos_simulation.get_err_rdiff_batch
        else:
            self.get_v_loss_r = None

        if strategy_config["r_loss_log_scale"]==True:
            self.r_loss_scale = self.log_loss_scale
        else:
            self.r_loss_scale = self.default_loss_scale

        self.run_eagerly = False

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")

        self.sample_gradient_sum_functions_list=None
    
    @tf.function
    def default_loss_scale(self, err_r, v_loss_r=None):
        loss_r = tf.math.reduce_sum(tf.math.square(err_r), axis=1)
        return loss_r, v_loss_r
    
    @tf.function
    def log_loss_scale(self, err_r, v_loss_r=None):
        err_r_quad = tf.math.reduce_sum(tf.math.square(err_r), axis=1)
        loss_r = tf.math.log(err_r_quad+1)
        if v_loss_r is not None:
            v_loss_r=tf.transpose(tf.transpose(v_loss_r)/(err_r_quad+1))
        return loss_r, v_loss_r
    
    @tf.function
    def get_v_loss_x(self, input, target_snapshot, snapshot_bound_batch):
        x_pred = self(input, training=True)
        x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,snapshot_bound_batch))
        v_loss_x = x_pred_denorm - target_snapshot  # We get the loss on the error for the denormalised snapshot
        return v_loss_x, x_pred_denorm
    
    @tf.function
    def get_gradients(self, trainable_vars, input_batch, snapshot_bound_batch, v_loss_x_batch, v_loss_r_batch, w_x, w_r):

        v_loss = 2*(w_x*v_loss_x_batch+w_r*v_loss_r_batch)
        print('w_x: ', w_x)
        print('w_r: ', w_r)

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(input_batch, training=True)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input_batch, snapshot_bound_batch))
            v_u_dotprod = tf.math.reduce_sum(tf.math.multiply(v_loss, x_pred_denorm), axis=1)
            v_u_dotprod_mean = tf.math.reduce_mean(v_u_dotprod)
        grad_loss=tape_d.gradient(v_u_dotprod_mean, trainable_vars)

        return grad_loss

    def generate_gradient_sum_functions(self):
        # @tf.function
        # def gradient_sum_sample(previous_gradients, gradients, batch_len):
        #     updated_gradients=previous_gradients+gradients/batch_len
        #     return updated_gradients
        
        # self.sample_gradient_sum_functions_list=[]
        # for i in range(len(self.trainable_variables)):
        #     self.sample_gradient_sum_functions_list.append(gradient_sum_sample)
        pass


    def train_step(self,data):
        input_batch, (target_snapshot_batch,target_aux_batch,snapshot_bound_batch) = data # target_aux is the reference force or residual, depending on the settings
        trainable_vars = self.trainable_variables

        v_loss_x_batch, x_pred_denorm_batch = self.get_v_loss_x(input_batch, target_snapshot_batch, snapshot_bound_batch)
        loss_x_batch = tf.math.reduce_sum(tf.math.square(v_loss_x_batch), axis=1)

        err_r_batch, v_loss_r_batch = self.get_v_loss_r(x_pred_denorm_batch,target_aux_batch)
        loss_r_batch, v_loss_r_batch = self.r_loss_scale(err_r_batch, v_loss_r_batch)

        total_loss_x=tf.math.reduce_mean(loss_x_batch)
        total_loss_r=tf.math.reduce_mean(loss_r_batch)

        grad_loss = self.get_gradients(trainable_vars, input_batch, snapshot_bound_batch, v_loss_x_batch, v_loss_r_batch, self.wx, self.wr)

        self.optimizer.apply_gradients(zip(grad_loss, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}
        # return total_loss_r*self.wr+total_loss_x*self.wx, grad_loss

    def test_step(self, data):
        input_batch, (target_snapshot_batch,target_aux_batch,snapshot_bound_batch) = data

        x_pred_batch = self(input_batch, training=False)
        x_pred_denorm_batch = self.prepost_processor.postprocess_output_data_tf(x_pred_batch, (input_batch, snapshot_bound_batch))
        err_x_batch = target_snapshot_batch - x_pred_denorm_batch
        loss_x_batch = tf.math.reduce_sum(tf.math.square(err_x_batch), axis=1)
        
        err_r_batch = self.get_err_r(x_pred_denorm_batch, target_aux_batch)
        loss_r_batch, _ = self.r_loss_scale(err_r_batch)

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