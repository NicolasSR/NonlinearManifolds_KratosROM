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
            self.get_v_loss_w = self.kratos_simulation.get_v_loss_wnorm
            self.get_r_w = self.kratos_simulation.get_r_wnorm
        elif strategy_config["r_loss_type"]=='diff':
            self.get_v_loss_w = self.kratos_simulation.get_v_loss_wdiff
            self.get_r_w = self.kratos_simulation.get_r_wdiff
        else:
            self.get_v_loss_w = None

        self.run_eagerly = False

        self.loss_w_tracker = keras.metrics.Mean(name="loss_w")

        self.sample_gradient_sum_functions_list=None
    
    @tf.function
    def w_loss_scale(self, err_w, v_loss_w=None):
        loss_w = err_w**2
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

            r_pred, v_loss_w = self.get_v_loss_w(x_pred_denorm,target_aux)
            err_w=tf.linalg.matmul(target_snapshot, target_aux, transpose_b=True)-tf.linalg.matmul(x_pred_denorm, r_pred, transpose_b=True)
            v_loss_w*=err_w
            loss_w, v_loss_w = self.w_loss_scale(err_w, v_loss_w)

            total_loss_w+=loss_w/batch_len

            grad_loss = self.get_gradients(trainable_vars, input, v_loss_w)

            for i in range(len(total_gradients)):
                total_gradients[i]=self.sample_gradient_sum_functions_list[i](total_gradients[i], grad_loss[i], batch_len)

        # self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        # self.loss_w_tracker.update_state(total_loss_w)
        # return {"loss_w": self.loss_w_tracker.result()}
        return total_loss_w, total_gradients

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

            r_pred = self.get_r_w(x_pred_denorm, target_aux)
            err_w=(tf.linalg.matmul(target_snapshot, target_aux, transpose_b=True)-tf.linalg.matmul(x_pred_denorm, r_pred, transpose_b=True))

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
    
    """ def test_gradients(self, data, crop_mat_tf, crop_mat_scp):
        import matplotlib.pyplot as plt

        input_batch, (target_snapshot_batch,target_aux_batch) = data

        sample_id=np.random.randint(0, input_batch.shape[0]-1)
        input=tf.expand_dims(input_batch[sample_id],axis=0)
        target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
        target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

        eps_vec = np.logspace(1, 12, 100)/1e13

        v=np.random.rand(1,crop_mat_scp.shape[1])
        v=v/np.linalg.norm(v)
        print(v)
        print(np.linalg.norm(v))

        err_vec=[]

        for eps in eps_vec:

            ev=v*eps

            # print(crop_mat_scp.shape)
            # print(ev.T.shape)
            ev_uncropped=(crop_mat_scp@ev.T).T
            x_pred = self(input, training=False)
            x_pred_denorm = self.prepost_processor.postprocess_output_data_tf(x_pred,input)
            x_app_denorm = x_pred_denorm+ev_uncropped

            r_pred, v_loss_w = self.get_v_loss_w(x_pred_denorm,target_aux)
            err_w=tf.linalg.matmul(target_snapshot, target_aux, transpose_b=True)-tf.linalg.matmul(x_pred_denorm, r_pred, transpose_b=True)
            v_loss_w*=err_w
            loss_w, v_loss_w = self.w_loss_scale(err_w, v_loss_w)

            v_loss = 2*v_loss_w
            v_I_dotprod = tf.linalg.matmul(v_loss, np.eye(v_loss.shape[1]))


            r_app = self.get_r_w(x_app_denorm, target_aux)
            err_w_app=(tf.linalg.matmul(target_snapshot, target_aux, transpose_b=True)-tf.linalg.matmul(x_app_denorm, r_app, transpose_b=True))
            loss_w_app, _ = self.w_loss_scale(err_w_app)

            lin_term=tf.linalg.matmul(v_I_dotprod, ev_uncropped, transpose_b=True).numpy()[0,0]
            # lin_term=0
            print(loss_w_app)
            print(loss_w)
            print(lin_term)
            err_vec.append(np.abs(loss_w_app.numpy()[0,0] - loss_w.numpy()[0,0] - lin_term))
        
        print(err_vec)
        square=np.power(eps_vec,2)
        plt.plot(eps_vec, square, "--", label="square")
        plt.plot(eps_vec, eps_vec, "--", label="linear")
        plt.plot(eps_vec, err_vec, label="error")
        # plt.plot(eps_vec, err_h, label="error_h")
        # plt.plot(eps_vec, err_l, label="error_l")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show() """

    def test_gradients(self, data, crop_mat_tf, crop_mat_scp):
        import matplotlib.pyplot as plt

        input_batch, (target_snapshot_batch,target_aux_batch) = data

        sample_id=np.random.randint(0, input_batch.shape[0]-1)
        input=tf.expand_dims(input_batch[sample_id],axis=0)
        target_snapshot=tf.expand_dims(target_snapshot_batch[sample_id],axis=0)
        target_aux=tf.expand_dims(target_aux_batch[sample_id],axis=0)

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
            # print(trainable_var)
            v_norm_check+=np.linalg.norm(trainable_var)**2
        print('Base noise L2 Norm: ', np.sqrt(v_norm_check))

        # print(v_norm)
        # print(v_list)

        eps_vec = np.logspace(1, 10, 100)/1e9

        err_vec=[]
        for eps in eps_vec:

            for i, trainable_var in enumerate(self.trainable_variables):
                trainable_var.assign(train_var_init_values[i])

            v_norm_check = 0.0
            for i, trainable_var in enumerate(self.trainable_variables):
                # print(trainable_var)
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
                # print(trainable_var)
                v_norm_check+=np.linalg.norm(trainable_var)**2
            print('Norm once noise is applied: ', np.sqrt(v_norm_check))

            # exit()

            # print(first_order_term)

            err_vec.append(np.abs(total_loss_w_ap.numpy()[0,0]-total_loss_w_o.numpy()[0,0]-first_order_term))

        square=np.power(eps_vec,2)
        plt.plot(eps_vec, square, "--", label="square")
        plt.plot(eps_vec, eps_vec, "--", label="linear")
        plt.plot(eps_vec, err_vec, label="error")
        # plt.plot(eps_vec, err_h, label="error_h")
        # plt.plot(eps_vec, err_l, label="error_l")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show()



            


