import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator

from ArchitectureFactories.PODANN_factory import PODANN_Architecture_Factory

from Utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, mean_l2_error, forbenius_error
from Utils.matrix_to_gid_output import output_GID_from_matrix

tf.keras.backend.set_floatx('float64')


if __name__ == "__main__":

    working_path=''
    model_path=working_path+'saved_models_cantilever_big_range/PODANN/PODANN_tf_sfarhat_diff_svd_Lay[200, 200]_Emb6.60_LRsgdr0.001/'

    with open(model_path+"train_config.npy", "rb") as train_config_file:
        train_config = np.load(train_config_file,allow_pickle='TRUE').item()
    print(train_config)
    dataset_path=working_path+train_config['dataset_path']

    # Select the architecture to use
    arch_factory = PODANN_Architecture_Factory(working_path, train_config["architecture"])

    # Create a fake Analysis stage to calculate the predicted residuals
    kratos_simulation = StructuralMechanics_KratosSimulator(working_path, train_config)
    crop_mat_tf, crop_mat_scp = kratos_simulation.get_crop_matrix()

    # Select the type of preprocessing (normalisation)
    prepost_processor=arch_factory.prepost_processor_selector(working_path, train_config["dataset_path"])

    S_test = np.load(working_path+train_config['dataset_path']+'S_test_small.npy')
    arch_factory.configure_prepost_processor(prepost_processor, S_test, crop_mat_tf, crop_mat_scp)

    S_out=prepost_processor.preprocess_nn_output_data(S_test)
    S_in, _=prepost_processor.preprocess_input_data(S_test)

    plt.boxplot(np.concatenate([S_in,S_out], axis=1))
    plt.show()

    print('======= Instantiating TF Model =======')
    network, _, __ = arch_factory.define_network(prepost_processor, kratos_simulation)
    
    print('======= Loading TF Model weights =======')
    network.load_weights(model_path+"model_weights.h5")
    
    Q_sup_pred = network(S_in).numpy()

    rel_err_features = np.abs(Q_sup_pred-S_out)/np.abs(S_out)

    plt.boxplot(rel_err_features)
    plt.show()