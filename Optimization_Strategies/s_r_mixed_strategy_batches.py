import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class S_R_Mixed_Strategy_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(S_R_Mixed_Strategy_KerasModel,self).__init__(*args,**kwargs)
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
        
        self.rescaling_factor_x = 1.0
        self.rescaling_factor_r = 1.0

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")

        self.sample_gradient_sum_functions_list=None
    
    @tf.function
    def default_loss_scale(self, err_r, v_loss_r=None):
        loss_r = tf.math.reduce_mean(tf.math.square(err_r), axis=1)
        return loss_r, v_loss_r
    
    @tf.function
    def log_loss_scale(self, err_r, v_loss_r=None):
        err_r_quad = tf.math.reduce_mean(tf.math.square(err_r), axis=1)
        loss_r = tf.math.log(err_r_quad+1)
        if v_loss_r is not None:
            v_loss_r=tf.transpose(tf.transpose(v_loss_r)/(err_r_quad+1))
        return loss_r, v_loss_r
    
    @tf.function
    def get_v_loss_x(self, input, target_snapshot):
        x_pred = self(input, training=True)
        x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input,None))
        v_loss_x = x_pred_denorm - target_snapshot  # We get the loss on the error for the denormalised snapshot
        return v_loss_x, x_pred_denorm
    
    @tf.function
    def get_gradients(self, trainable_vars, input, v_loss_x, v_loss_r, w_x, w_r):

        v_loss = 2*(w_x*v_loss_x/self.rescaling_factor_x+w_r*v_loss_r/self.rescaling_factor_r)

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(input, training=True)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input, None))
            v_u_dotprod = tf.math.reduce_mean(tf.math.multiply(v_loss, x_pred_denorm), axis=1)
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

    def update_rescaling_factors(self, S_true, R_true):

        S_recons_aux1 = self.prepost_processor.preprocess_nn_output_data(S_true)
        S_recons_aux2, _ =self.prepost_processor.preprocess_input_data(S_true)
        S_recons = self.prepost_processor.postprocess_output_data(np.zeros(S_recons_aux1.shape), (S_recons_aux2, None))

        rescaling_factor_x = np.mean(np.square(S_recons-S_true))
        
        err_r_batch, _ = self.get_v_loss_r(tf.constant(S_recons), tf.constant(R_true))
        rescaling_factor_r = np.mean(np.square(err_r_batch.numpy()))

        self.rescaling_factor_x = rescaling_factor_x
        self.rescaling_factor_r = rescaling_factor_r

        print('Updated gradient rescaling factors. x: ' + str(self.rescaling_factor_x) + ', r: ' + str(self.rescaling_factor_r))

    def train_step(self,data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data # target_aux is the reference force or residual, depending on the settings
        trainable_vars = self.trainable_variables

        v_loss_x_batch, x_pred_denorm_batch = self.get_v_loss_x(input_batch, target_snapshot_batch)
        loss_x_batch = tf.math.reduce_mean(tf.math.square(v_loss_x_batch), axis=1)

        err_r_batch, v_loss_r_batch = self.get_v_loss_r(x_pred_denorm_batch,target_aux_batch)
        loss_r_batch, v_loss_r_batch = self.r_loss_scale(err_r_batch, v_loss_r_batch)

        total_loss_x=tf.math.reduce_mean(loss_x_batch)
        total_loss_r=tf.math.reduce_mean(loss_r_batch)

        grad_loss = self.get_gradients(trainable_vars, input_batch, v_loss_x_batch, v_loss_r_batch, self.wx, self.wr)
        
        # for i, grad in enumerate(grad_loss):
        #     tf.print()
        #     tf.print(tf.reduce_max(tf.math.abs(grad_loss[i])))
        #     tf.print(tf.reduce_mean(tf.math.abs(grad_loss[i])))
        #     tf.print(tf.reduce_min(tf.math.abs(grad_loss[i])))
        #     tf.print(self.rescaling_factor_r)

        self.optimizer.apply_gradients(zip(grad_loss, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}
        # return total_loss_r*self.wr+total_loss_x*self.wx, grad_loss

    def test_step(self, data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        x_pred_batch = self(input_batch, training=False)
        x_pred_denorm_batch = self.prepost_processor.postprocess_output_data_tf(x_pred_batch, (input_batch, None))
        err_x_batch = target_snapshot_batch - x_pred_denorm_batch
        loss_x_batch = tf.math.reduce_mean(tf.math.square(err_x_batch), axis=1)
        
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
    
    
    def test_gradients(self, data, crop_mat_tf, crop_mat_scp):
        import matplotlib.pyplot as plt

        self.wr=0.001
        self.wx=0.001

        input_batch, (target_snapshot_batch,target_aux_batch) = data

        # sample_id=np.random.randint(0, input_batch.shape[0]-1, size=20)
        sample_id=[0,1,2,3,4,5,6,7,8,9]
        input=input_batch[sample_id]
        target_snapshot=target_snapshot_batch[sample_id]
        target_aux=target_aux_batch[sample_id]

        v_list=[]
        train_var_init_values=[]
        v_norm=0.0
        for trainable_var in self.trainable_variables:
            v=np.random.rand(*trainable_var.shape.as_list())
            v_norm+=np.linalg.norm(v)**2
            v_list.append(v)
            train_var_init_values.append(tf.identity(trainable_var))
        for v in v_list:
            v/=np.sqrt(v_norm)

        v_norm_check = 0.0
        for i, trainable_var in enumerate(v_list):
            v_norm_check+=np.linalg.norm(trainable_var)**2
        print('Base noise L2 Norm: ', np.sqrt(v_norm_check))

        eps_vec = np.logspace(1, 10, 100)/1e9

        err_vec=[]
        for eps in eps_vec:

            for i, trainable_var in enumerate(self.trainable_variables):
                trainable_var.assign(train_var_init_values[i])

            v_norm_check = 0.0
            for i, trainable_var in enumerate(self.trainable_variables):
                v_norm_check+=np.linalg.norm(trainable_var)**2
            print('Init step norm: ', np.sqrt(v_norm_check))

            total_loss_w_o, total_gradients_o = self.train_step((input,(target_snapshot,target_aux)))

            for i, trainable_var in enumerate(self.trainable_variables):
                trainable_var.assign_add(v_list[i]*eps)
            
            total_loss_w_ap, _ = self.train_step((input,(target_snapshot,target_aux)))

            first_order_term=0.0
            for i, gradient_o in enumerate(total_gradients_o):
                first_order_term+=np.sum(np.multiply(gradient_o, v_list[i]*eps))

            v_norm_check = 0.0
            for i, v in enumerate(v_list):
                v_norm_check+=np.linalg.norm(v*eps)**2
            print('Noise norm: ', np.sqrt(v_norm_check))

            v_norm_check = 0.0
            for i, trainable_var in enumerate(self.trainable_variables):
                v_norm_check+=np.linalg.norm(trainable_var)**2
            print('Norm once noise is applied: ', np.sqrt(v_norm_check))

            err_vec.append(np.abs(total_loss_w_ap.numpy()-total_loss_w_o.numpy()-first_order_term))


        print(err_vec)

        square=np.power(eps_vec,2)
        plt.plot(eps_vec, square, "--", label="square")
        plt.plot(eps_vec, eps_vec, "--", label="linear")
        plt.plot(eps_vec, err_vec, label="error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show()
