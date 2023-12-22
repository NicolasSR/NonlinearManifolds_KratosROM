import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class R_Only_Strategy_KerasModel(keras.Model):

    def __init__(self, prepost_processor, kratos_simulation, strategy_config, *args, **kwargs):
        super(R_Only_Strategy_KerasModel,self).__init__(*args,**kwargs)

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
    def get_gradients(self, trainable_vars, input_batch, v_loss_r_batch):

        v_loss = 2*v_loss_r_batch

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(input_batch, training=True)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred, (input_batch,None))
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
        input_batch, (target_snapshot_batch,target_aux_batch) = data # target_aux is the reference force or residual, depending on the settings
        trainable_vars = self.trainable_variables

        x_pred_batch = self(input_batch, training=True)
        x_pred_denorm_batch = self.prepost_processor.postprocess_output_data_tf(x_pred_batch,(input_batch,None))
        err_x_batch = target_snapshot_batch - x_pred_denorm_batch
        loss_x_batch = tf.math.reduce_sum(tf.math.square(err_x_batch), axis=1)

        err_r_batch, v_loss_r_batch = self.get_v_loss_r(x_pred_denorm_batch,target_aux_batch)
        loss_r_batch, v_loss_r_batch = self.r_loss_scale(err_r_batch, v_loss_r_batch)

        total_loss_x=tf.math.reduce_mean(loss_x_batch)
        total_loss_r=tf.math.reduce_mean(loss_r_batch)

        grad_loss = self.get_gradients(trainable_vars, input_batch, v_loss_r_batch)

        # for i, grad in enumerate(grad_loss):
        #     grad_loss[i] = grad*100000000
            # tf.print(tf.reduce_max(grad_loss[i]))
            # tf.print(tf.reduce_mean(grad_loss[i]))

        self.optimizer.apply_gradients(zip(grad_loss, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}
        # return total_loss_r, grad_loss

    def test_step(self, data):
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        x_pred_batch = self(input_batch, training=False)
        x_pred_denorm_batch = self.prepost_processor.postprocess_output_data_tf(x_pred_batch,(input_batch,None))
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


        square=np.power(eps_vec,2)
        plt.plot(eps_vec, square, "--", label="square")
        plt.plot(eps_vec, eps_vec, "--", label="linear")
        plt.plot(eps_vec, err_vec, label="error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show()

    
    def test_gradients_jacobian(self, data, crop_mat_tf, crop_mat_scp):
        import matplotlib.pyplot as plt
        import scipy

        # input_batch, (target_snapshot_batch,target_aux_batch) = data
        input_batch, (target_snapshot_batch,target_aux_batch) = data

        sample_id=np.random.randint(0, input_batch.shape[0]-1)

        target_snapshot=np.expand_dims(target_snapshot_batch[sample_id],axis=0)
        target_aux=np.expand_dims(target_aux_batch[sample_id],axis=0)

        print(crop_mat_scp.shape)
        eye=np.eye(crop_mat_scp.shape[0])

        v=np.random.rand(*target_snapshot.shape)
        v=((eye-crop_mat_scp)@v.T).T
        v=v/np.linalg.norm(v)

        eps_vec = np.logspace(1, 10, 100)/1e9

        b_target=self.kratos_simulation.get_r_batch_(target_snapshot).copy()
        A_u0=self.kratos_simulation.get_A_(target_snapshot[0]).copy()

        # print(A_u0)
        Jac_err=np.random.rand(*A_u0.shape)

        print('YAY')


        err_vec=[]
        for eps in eps_vec:
            
            v_eps=v*eps
            u_approx=target_snapshot+v_eps
            
            b_approx=self.kratos_simulation.get_r_batch_(u_approx).copy()

            # A_e_vec = self.kratos_simulation.get_A_e_vec_batch_(target_snapshot, v_eps).copy()


            # A_u0=self.kratos_simulation.get_A_(target_snapshot[0]).copy()
            # err=b_approx-b_target+(A_u0@v_eps.T).T
            Jac_err=Jac_err/np.linalg.norm(Jac_err)*1e6
            err=b_approx-b_target+((A_u0+Jac_err)@v_eps.T).T
            # err=b_approx-b_target+((Jac_err)@v_eps.T).T
        
            # err=b_approx-b_target+A_e_vec

            

            err_vec.append(np.linalg.norm(err))


        square=np.power(eps_vec,2)
        plt.plot(eps_vec, square, "--", label="square")
        plt.plot(eps_vec, eps_vec, "--", label="linear")
        plt.plot(eps_vec, err_vec, label="error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show()

