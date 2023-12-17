import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class W_Only_Strategy_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(W_Only_Strategy_KerasModel,self).__init__(*args,**kwargs)

        self.prepost_processor = prepost_processor
        self.kratos_simulation = kratos_simulation

        if strategy_config["r_loss_type"]=='norm':
            self.get_v_loss_w = self.kratos_simulation.get_v_loss_wdiffnorm
            self.get_err_w = self.kratos_simulation.get_err_wdiffnorm
        elif strategy_config["r_loss_type"]=='diff':
            self.get_v_loss_w = self.kratos_simulation.get_v_loss_wdiffdiff
            self.get_err_w = self.kratos_simulation.get_err_wdiffdiff
        else:
            self.get_v_loss_w = None

        self.run_eagerly = False

        self.loss_w_tracker = keras.metrics.Mean(name="loss_w")

        self.sample_gradient_sum_functions_list=None
    
    @tf.function
    def w_loss_scale(self, err_w, v_loss_w=None):
        loss_w = tf.linalg.matmul(err_w,err_w,transpose_b=True)
        return loss_w, v_loss_w
    
    @tf.function
    def get_gradients(self, trainable_vars, input, v_loss_w):

        v_loss = 2*v_loss_w

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
        
        total_loss_w = 0

        for sample_id in range(batch_len):
            
            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

            x_pred = self(input, training=True)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred,input)

            err_w, v_loss_w = self.get_v_loss_w(x_pred_denorm,target_snapshot,target_aux)
            loss_w, v_loss_w = self.w_loss_scale(err_w, v_loss_w)

            total_loss_w+=loss_w/batch_len

            grad_loss = self.get_gradients(trainable_vars, input, v_loss_w)

            for i in range(len(total_gradients)):
                total_gradients[i]=self.sample_gradient_sum_functions_list[i](total_gradients[i], grad_loss[i], batch_len)

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        self.loss_w_tracker.update_state(total_loss_w)
        return {"loss_w": self.loss_w_tracker.result()}

    def test_step(self, data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        batch_len=input_batch.shape[0]

        total_loss_w = 0

        for sample_id in range(batch_len):

            input=tf.expand_dims(input_batch[sample_id],axis=0)
            target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
            target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

            x_pred = self(input, training=False)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred,input)

            err_w = self.get_err_w(x_pred_denorm, target_snapshot, target_aux)

            loss_w, _ = self.w_loss_scale(err_w)

            total_loss_w+=loss_w/batch_len

        # Compute our own metrics
        self.loss_w_tracker.update_state(total_loss_w)
        return {"loss_w": self.loss_w_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        
        return [self.loss_w_tracker]
