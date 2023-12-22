import numpy as np
import os
import json

import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.optimizers import AdamW, SGD
from keras.callbacks import LearningRateScheduler



if __name__=="__main__":

    model_name='test_gradients'
    n_inf=6
    n_sup=60
    layers_size=[200,200]
    batch_size=16
    epochs=200

    working_path=''
    model_path=working_path+'saved_models_cantilever_big_range/PODANN_Standalone/'+model_name+'/'
    dataset_path='datasets_rubber_hyperelastic_cantilever_big_range/'

    os.makedirs(model_path, exist_ok=False)

    S_train = np.load(dataset_path+'S_train.npy')
    # S_train = S_train[[0]]
    S_val = np.load(dataset_path+'S_val.npy')

    print('S_train: ', S_train)

    phi = np.load(dataset_path+'POD/phi.npy')

    phi_inf=phi[:,:n_inf].copy()
    phi_sup=phi[:,n_inf:n_sup].copy()
    print('Phi_inf matrix shape: ', phi_inf.shape)
    print('Phi_sgs matrix shape: ', phi_sup.shape)

    Q_inf_train = (phi_inf.T@S_train.T).T
    Q_sup_train = (phi_sup.T@S_train.T).T
    Q_inf_val = (phi_inf.T@S_val.T).T
    Q_sup_val = (phi_sup.T@S_val.T).T
    print('Q_inf_train matrix shape: ', Q_inf_train.shape)
    print('Q_sup_train matrix shape: ', Q_sup_train.shape)
    print('Q_inf_val matrix shape: ', Q_inf_val.shape)
    print('Q_sup_val matrix shape: ', Q_sup_val.shape)

    print('Q_inf_train: ', Q_inf_train)
    print('Q_sup_train: ', Q_sup_train)

    input_layer=layers.Input((n_inf,))
    layer_out=input_layer
    for size in layers_size:
        layer_out=layers.Dense(size, 'elu', use_bias=False, kernel_initializer="he_normal")(layer_out)
    output_layer=layers.Dense(n_sup-n_inf, 'linear', use_bias=False, kernel_initializer="he_normal")(layer_out)

    network=Model(input_layer, output_layer)
    network.compile(AdamW(0.001), loss='mse', run_eagerly=False)
    # network.compile(SGD(0.001), loss='mse', run_eagerly=False)
    network.summary()

    # print('======= Loading saved weights =======')
    # network.load_weights(working_path+'saved_models_cantilever_big_range/PODANN_Standalone/test_sfarhat_lrsgdr/model_weights.h5')

    def lr_scheduler(epoch, lr):
        if epoch>200:
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
    
    # callbacks = [LearningRateScheduler(lr_sgdr_schedule, verbose=1)]

    print('======= Training model =======')
    # history = network.fit(Q_inf_train, Q_sup_train, batch_size=batch_size, epochs=epochs, validation_data=(Q_inf_val,Q_sup_val), shuffle=True, callbacks=callbacks)
    history = network.fit(Q_inf_train, Q_sup_train, batch_size=batch_size, epochs=epochs, validation_data=(Q_inf_val,Q_sup_val), shuffle=True)

    train_config={
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'PODANN', # ['POD','Quad','PODANN]
            "q_inf_size": 6,
            "q_sup_size": 60,
            "hidden_layers": [200,200],
            "prepost_process": 'svd',
            "opt_strategy": {
                "name": 'tf_sfarhat', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
                "r_loss_type": 'diff',  # ['norm, 'diff']
                "r_loss_log_scale": False,
                "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
                "batch_size": batch_size,
                "epochs": epochs
            },
            "finetune_from": 'saved_models_cantilever_big_range/PODANN_Standalone/test_sfarhat_lrsgdr/',
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