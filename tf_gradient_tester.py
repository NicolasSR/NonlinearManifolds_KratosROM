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
from ArchitectureFactories.Quad_factory import Quad_Architecture_Factory
from ArchitectureFactories.POD_factory import POD_Architecture_Factory

from Utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, mean_l2_error, forbenius_error
from Utils.matrix_to_gid_output import output_GID_from_matrix

tf.keras.backend.set_floatx('float64')

class TF_Gradient_Tester():

    def __init__(self, working_path, model_path):
        self.working_path=working_path
        self.model_path=working_path+model_path

        self.model_weights_path=self.model_path
        self.model_weights_filename='model_weights.h5'

        with open(self.model_path+"train_config.npy", "rb") as train_config_file:
            self.train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        print(self.train_config)
        self.dataset_path=working_path+self.train_config['dataset_path']

    def prepare_evaluation_data(self, ):

        if self.test_validation==False:
            S_test=np.load(self.dataset_path+'S_test.npy')
            R_test=np.load(self.dataset_path+'R_test.npy')
            F_test=np.load(self.dataset_path+'F_test.npy')
        else:
            S_test=np.load(self.dataset_path+'S_val.npy')
            R_test=np.load(self.dataset_path+'R_val.npy')
            F_test=np.load(self.dataset_path+'F_val.npy')

        return S_test, R_test, F_test

    def architecture_factory_selector(self, arch_config):
        arch_type = arch_config["name"]
        if arch_type == 'PODANN':
            return PODANN_Architecture_Factory(self.working_path, arch_config)
        elif arch_type == 'Quad':
            return Quad_Architecture_Factory(self.working_path, arch_config)
        elif arch_type == 'POD': 
            return POD_Architecture_Factory(self.working_path, arch_config)
        else:
            print('No valid architecture type was selected')
            return None
        
    def kratos_simulator_selector(self, sim_type):
        # if 'fluid' in sim_type:
        #     return KratosSimulator_Fluid
        # else:
        return StructuralMechanics_KratosSimulator
    
    def get_orig_fom_snapshots(self):
        S_FOM_orig=np.load(self.working_path+self.train_config['dataset_path']+'FOM.npy')
        return S_FOM_orig

    def execute_test(self):

        # Select the architecture to use
        arch_factory = self.architecture_factory_selector(self.train_config["architecture"])

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation_class=self.kratos_simulator_selector(self.train_config["sim_type"])
        self.kratos_simulation = kratos_simulation_class(self.working_path, self.train_config)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        # Select the type of preprocessing (normalisation)
        self.prepost_processor=arch_factory.prepost_processor_selector(self.working_path, self.train_config["dataset_path"])

        S_FOM_orig = self.get_orig_fom_snapshots()
        arch_factory.configure_prepost_processor(self.prepost_processor, S_FOM_orig, crop_mat_tf, crop_mat_scp)

        print('======= Instantiating TF Model =======')
        self.network, _, __ = arch_factory.define_network(self.prepost_processor, self.kratos_simulation)
        
        print('======= Loading TF Model weights =======')
        self.network.load_weights(self.model_weights_path+self.model_weights_filename)


        input_data, target_data, val_input, val_target = arch_factory.get_training_data(self.prepost_processor, self.train_config["dataset_path"])

        # S_test, R_test, F_test = self.prepare_evaluation_data()
        # print('Shape S_test: ', S_test.shape)
        # print('Shape R_test: ', R_test.shape)
        # print('Shape F_test: ', F_test.shape)

        self.network.test_gradients((input_data, target_data), crop_mat_tf, crop_mat_scp)