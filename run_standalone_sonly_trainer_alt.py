import numpy as np
import os
import json

import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.optimizers import AdamW, SGD
from keras.callbacks import LearningRateScheduler

from matplotlib import pyplot as plt



if __name__=="__main__":

    model_name='test_sonly_gradients_adamw'
    n_inf=6
    n_sup=60
    layers_size=[200,200]
    batch_size=16
    epochs=200
    lr=0.001

    working_path=''
    model_path=working_path+'saved_models_cantilever_big_range/PODANN_Standalone/'+model_name+'/'
    dataset_path='datasets_rubber_hyperelastic_cantilever_big_range/'

    os.makedirs(model_path, exist_ok=False)

    S_train = np.load(dataset_path+'S_train.npy')
    S_val = np.load(dataset_path+'S_val.npy')

    # S_train=S_train[:14]

    phi = np.load(dataset_path+'PODANN/phi_whitenostand.npy')
    sigma_vec = np.load(dataset_path+'PODANN/sigma_whitenostand.npy')

    phi_inf=phi[:,:n_inf].copy()
    phi_sup=phi[:,n_inf:n_sup].copy()
    print('Phi_inf matrix shape: ', phi_inf.shape)
    print('Phi_sgs matrix shape: ', phi_sup.shape)

    sigma_vec_inf=sigma_vec[:n_inf].copy()
    sigma_vec_sup=sigma_vec[n_inf:n_sup].copy()

    sigma_inf=np.diag(sigma_vec_inf)
    sigma_sup=np.diag(sigma_vec_sup)
    sigma_inv_inf=np.linalg.inv(sigma_inf)
    sigma_inv_sup=np.linalg.inv(sigma_sup)
    print('sigma_inf matrix shape: ', sigma_inf.shape)
    print('sigma_sup matrix shape: ', sigma_sup.shape)
    print('sigma_inv_inf matrix shape: ', sigma_inv_inf.shape)
    print('sigma_inv_sup matrix shape: ', sigma_inv_sup.shape)

    Q_inf_train = (sigma_inv_inf@phi_inf.T@S_train.T).T
    Q_sup_train = (sigma_inv_sup@phi_sup.T@S_train.T).T
    Q_inf_val = (sigma_inv_inf@phi_inf.T@S_val.T).T
    Q_sup_val = (sigma_inv_sup@phi_sup.T@S_val.T).T
    print('Q_inf_train matrix shape: ', Q_inf_train.shape)
    print('Q_sup_train matrix shape: ', Q_sup_train.shape)
    print('Q_inf_val matrix shape: ', Q_inf_val.shape)
    print('Q_sup_val matrix shape: ', Q_sup_val.shape)

    # target_train=(S_train.T-phi_inf@sigma_inf@Q_inf_train.T).T
    # target_val=(S_val.T-phi_inf@sigma_inf@Q_inf_val.T).T
    # print('target_train matrix shape: ', target_train.shape)
    # print('target_val matrix shape: ', target_val.shape)

    n_dofs=S_train.shape[1]

    # input_layer=layers.Input((n_dofs,), dtype=tf.float64)
    # layer_out=layers.Lambda(lambda x: tf.transpose(tf.matmul(sigma_inv_inf,tf.linalg.matmul(phi_inf,x,transpose_a=True,transpose_b=True))), dtype=tf.float64)(input_layer)
    input_layer=layers.Input((n_inf,), dtype=tf.float64)
    layer_out=input_layer
    for size in layers_size:
        layer_out=layers.Dense(size, 'elu', use_bias=False, kernel_initializer="he_normal")(layer_out)
    layer_out=layers.Dense(n_sup-n_inf, 'linear', use_bias=False, kernel_initializer="he_normal")(layer_out)
    output_layer_aux_1=layers.Lambda(lambda x: tf.transpose(tf.matmul(phi_sup,tf.linalg.matmul(sigma_sup,x,transpose_b=True))), dtype=tf.float64)(layer_out)
    output_layer_aux_2=layers.Lambda(lambda x: tf.transpose(tf.matmul(phi_inf,tf.linalg.matmul(sigma_inf,x,transpose_b=True))), dtype=tf.float64)(input_layer)
    output_layer=layers.Add()([output_layer_aux_1, output_layer_aux_2])


    network=Model(input_layer, output_layer)
    network.compile(AdamW(lr), loss='mse', run_eagerly=False)
    # network.compile(SGD(lr), loss='mse', run_eagerly=False)
    network.summary()

    print('======= Loading saved weights =======')
    network.load_weights(working_path+'saved_models_cantilever_big_range/PODANN/test_sonly_gradients_base/model_weights.h5')

    def lr_scheduler(epoch, lr):
        if epoch>150:
            lr=0.0001
        # decay_rate = 0.85
        # decay_step = 1
        # if epoch % decay_step == 0 and epoch:
        #     return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr
    
    def lr_sgdr_schedule(epoch, lr):
        """
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
        """
        if epoch==0:
            new_lr=0.001
        else:
            max_lr=0.001
            min_lr=1e-6
            cycle_length=200
            scale_factor=10
            cycle=np.floor(epoch/cycle_length)
            x=np.abs(epoch/cycle_length-cycle)
            new_lr = min_lr+(max_lr-min_lr)*0.5*(1+np.cos(x*np.pi))/scale_factor**cycle
        return new_lr
    
    callbacks = [LearningRateScheduler(lr_sgdr_schedule, verbose=1)]

    # history = network.fit(S_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(S_val,target_val), shuffle=True, callbacks=callbacks)
    history = network.fit(Q_inf_train, S_train, batch_size=batch_size, epochs=epochs, validation_data=(Q_inf_val,S_val), shuffle=False, validation_batch_size=1)

    train_config={
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'PODANN', # ['POD','Quad','PODANN]
            "q_inf_size": 6,
            "q_sup_size": 60,
            "hidden_layers": [200,200],
            "prepost_process": 'svd_white_nostand',
            "opt_strategy": {
                "name": 'tf_sonly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
                "r_loss_type": 'diff',  # ['norm, 'diff']
                "r_loss_log_scale": False,
                "learning_rate": ('const', lr), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
                "batch_size": batch_size,
                "epochs": epochs
            },
            "finetune_from": None,
            "augmented": False,
            "use_bias": False,
            "use_dropout": None
        },
        "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
        "models_path_root": 'saved_models_cantilever_big_range/',
        "project_parameters_file":'ProjectParameters_tf.json'
   }
    
    print('======= Saving AE Config =======')
    with open(model_path+"train_config.npy", "wb") as ae_config_file:
        np.save(ae_config_file, train_config)
    with open(model_path+"train_config.json", "w") as ae_config_json_file:
        json.dump(train_config, ae_config_json_file)

    print('=========== Saving weights and history ============')
    network.save_weights(model_path+"model_weights.h5")
    with open(model_path+"history.json", "w") as history_file:
        json.dump(history.history, history_file)