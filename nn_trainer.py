import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np
import scipy

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator
from Kratos_Simulators.fluid_dynamics_kratos_simulator import FluidDynamics_KratosSimulator

from ArchitectureFactories.PODANN_factory import PODANN_Architecture_Factory
from ArchitectureFactories.Quad_factory import Quad_Architecture_Factory
from ArchitectureFactories.POD_factory import POD_Architecture_Factory

import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

class NN_Trainer():
    def __init__(self,working_path,train_config):
        self.working_path=working_path
        self.train_config=train_config

    def setup_output_directory(self, arch_factory):
        
        arch_folder_name, model_name = arch_factory.generate_model_name_part()
        if self.train_config["name"] is None:
            self.model_path=self.working_path+self.train_config["models_path_root"]
            self.model_path+=arch_folder_name+'/'
            self.model_path+=model_name
        else:
            self.model_path=self.working_path+self.train_config["models_path_root"]+self.train_config["name"]

        while os.path.isdir(self.model_path+'/'):
            self.model_path+='_bis'
        self.model_path+='/'

        print('Created Model directory at: ', self.model_path)
        
        os.makedirs(self.model_path, exist_ok=False)
        os.makedirs(self.model_path+"best/", exist_ok=True)
        os.makedirs(self.model_path+"last/", exist_ok=True)

        return self.model_path
    
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
        if sim_type=="structural":
            return StructuralMechanics_KratosSimulator
        elif sim_type=='fluid':
            return FluidDynamics_KratosSimulator
        return StructuralMechanics_KratosSimulator
    
    def execute_training(self):

        # Select the architecture to use
        arch_factory = self.architecture_factory_selector(self.train_config["architecture"])

        actual_model_path = self.setup_output_directory(arch_factory)
        arch_factory.set_actual_model_path(actual_model_path)

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation_class=self.kratos_simulator_selector(self.train_config["sim_type"])
        kratos_simulation = kratos_simulation_class(self.working_path, self.train_config)
        crop_mat_tf, crop_mat_scp = kratos_simulation.get_crop_matrix()

        # Select the type of preprocessimg (normalisation)
        prepost_processor=arch_factory.prepost_processor_selector(self.working_path, self.train_config["dataset_path"])

        S_FOM_orig = arch_factory.get_orig_fom_snapshots(self.train_config['dataset_path'])
        arch_factory.configure_prepost_processor(prepost_processor, S_FOM_orig, crop_mat_tf, crop_mat_scp)
        
        # nn_output_data = prepost_processor.preprocess_nn_output_data(S_FOM_orig)
        # print('Shape NN_output_data: ', nn_output_data.shape)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        network, _, __ = arch_factory.define_network(prepost_processor, kratos_simulation)
        
        if not self.train_config["architecture"]["finetune_from"] is None:
            print('======= Loading saved weights =======')
            network.load_weights(self.working_path+self.train_config["architecture"]["finetune_from"]+'model_weights.h5')

        # Get training data
        print('======= Loading training data =======')
        input_data, target_data, val_input, val_target = prepost_processor.get_training_data(self.train_config["architecture"])

        print('Shape input_data:', input_data.shape)
        for i in range(len(target_data)):
            print('Shape target_data [', i, ']: ', target_data[i].shape)
        print('Shape val_input:', val_input.shape)
        for i in range(len(val_target)):
            print('Shape target_data [', i, ']: ', val_target[i].shape)

        # input_data=input_data[:14]
        # target_data=list(target_data)
        # target_data[0]=target_data[0][:14]
        # target_data[1]=target_data[1][:14]

        print('input data: ', input_data)
        print('target_data: ', target_data)

        print('======= Saving AE Config =======')
        with open(self.model_path+"train_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, self.train_config)
        with open(self.model_path+"train_config.json", "w") as ae_config_json_file:
            json.dump(self.train_config, ae_config_json_file)

        print(self.train_config)

        print('=========== Starting training routine ============')
        history = arch_factory.train_network(network, input_data, target_data, val_input, val_target)

        # np.save(self.model_path+"gradients.npy", np.array(network.gradients_max, network.gradients_mean))

        print('=========== Saving weights and history ============')
        network.save_weights(self.model_path+"model_weights.h5")
        with open(self.model_path+"history.json", "w") as history_file:
            json.dump(history.history, history_file)

        print(self.train_config)

        # Dettach the fake sim (To prevent problems saving the model)
        network.kratos_simulation = None
        
    