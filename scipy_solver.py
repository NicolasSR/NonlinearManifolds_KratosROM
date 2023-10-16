import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator

from ArchitectureFactories.POD_factory import POD_Architecture_Factory
from ArchitectureFactories.Quad_factory import Quad_Architecture_Factory
from ArchitectureFactories.PODANN_factory import PODANN_Architecture_Factory

tf.keras.backend.set_floatx('float64')
 
class Scipy_Solver():

    def __init__(self, working_path, model_path, best):
        self.working_path=working_path
        self.model_path=working_path+model_path
        self.results_path=self.model_path+'scipy_solver_results/'
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx_'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr_'
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
            self.best_name_part=''
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        with open(self.model_path+"train_config.npy", "rb") as train_config_file:
            self.train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        print(self.train_config)
        self.dataset_path=working_path+self.train_config['dataset_path']

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

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
    
    def prepare_reference_data(self):
        S_true=np.load(self.dataset_path+'FOM/FOM_equalForces_30steps.npy')
        F_true=np.load(self.dataset_path+'FOM/POINTLOADS_equalForces_30steps.npy')
        return S_true, F_true
    
    def optimisation_routine(self, q0, f_vectors, s_true):
        f_vectors=np.expand_dims(f_vectors, axis=0)
        snapshot_true=np.expand_dims(s_true, axis=0)

        print()
        
        q_goal=self.encode(snapshot_true)
        S_goal_pred=self.decode(q_goal)

        r_vector_goal = self.kratos_simulation.get_r_forces_(S_goal_pred, f_vectors)[0]
        goal_r_norm = np.linalg.norm(r_vector_goal)
        print('Goal residual norm: ', goal_r_norm)

        def opt_function(x):
            if len(x.shape)==1:
                x=np.expand_dims(x,axis=0)
            snapshot = self.decode(x)
            r_vector = self.kratos_simulation.get_r_forces_(snapshot, f_vectors)[0]
            r_norm = np.linalg.norm(r_vector)
            return r_norm
        
        def opt_function_u_diff(x):
            if len(x.shape)==1:
                x=np.expand_dims(x,axis=1)
            snapshot = self.data_normalizer.process_input_to_raw_format(x)
            s_norm = np.linalg.norm(snapshot_true - snapshot)
            return s_norm
        
        # minimizer_kwargs = {"method":"L-BFGS-B", "options":{"maxiter":2}}
        # q_optim = scipy.optimize.basinhopping(opt_function, q0, niter=2, minimizer_kwargs=minimizer_kwargs)
        optim_result = scipy.optimize.basinhopping(opt_function, q0)

        print(optim_result)

        if len(optim_result.x.shape)==1:
            q_final=np.expand_dims(optim_result.x,axis=0)
        else:
            q_final=optim_result.x
        
        snapshot_final = self.decode(q_final).numpy()
        snapshot_rel_error_to_FOM = np.linalg.norm(snapshot_final-snapshot_true)/np.linalg.norm(snapshot_true)
        
        r_vector_final = self.kratos_simulation.get_r_forces_(snapshot_final, f_vectors)
        final_r_norm = np.linalg.norm(r_vector_final)

        print('Final r norm: ', final_r_norm)
        print('Relative error on x: ', snapshot_rel_error_to_FOM)

        reactions = -1*self.kratos_simulation.get_r_forces_withDirich_(snapshot_final, f_vectors)

        iteration_errors_list = [snapshot_rel_error_to_FOM,final_r_norm,goal_r_norm]
        
        return optim_result, snapshot_final, reactions, iteration_errors_list

    def execute_solver(self):

        # Select the architecture to use
        arch_factory = self.architecture_factory_selector(self.train_config["architecture"])

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation_class=self.kratos_simulator_selector(self.train_config["sim_type"])
        self.kratos_simulation = kratos_simulation_class(self.working_path, self.train_config)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        # Select the type of preprocessing (normalisation)
        prepost_processor=arch_factory.prepost_processor_selector(self.working_path, self.train_config["dataset_path"])

        S_FOM_orig = self.get_orig_fom_snapshots()
        arch_factory.configure_prepost_processor(prepost_processor, S_FOM_orig, crop_mat_tf, crop_mat_scp)

        print('======= Instantiating TF Model =======')
        network, enc_network, dec_network = arch_factory.define_network(prepost_processor, self.kratos_simulation)
        
        print('======= Loading TF Model weights =======')
        network.load_weights(self.model_weights_path+self.model_weights_filename)

        print('======= Defining encoder and decoder routines =======')
        self.encode=arch_factory.NMROM_encoder(prepost_processor, enc_network)
        self.decode=arch_factory.NMROM_encoder(prepost_processor, dec_network)

        print('======= Preparing reference data =======')
        S_true, F_true = self.prepare_reference_data()
        print('S_true shape: ', S_true)
        print('F_true shape: ', F_true)

        print('======= Running solver routine =======')
        results_file_path = self.results_path+'Scipy_results_matrix.npy'
        snapshots_file_path = self.results_path+'Scipy_snapshots_matrix.npy'
        reactions_file_path = self.results_path+'Scipy_reactions_matrix.npy'

        np.save(results_file_path, np.array([]))
        np.save(snapshots_file_path, np.array([]))
        np.save(reactions_file_path, np.array([]))

        q0 = np.zeros(6)

        for i, forces in enumerate(F_true):

            optim_result, snapshot_final, reactions, iteration_errors_list   = self.optimisation_routine(q0, forces, S_true[i])

            results_mat = list(np.load(results_file_path))
            results_mat.append(iteration_errors_list)
            np.save(results_file_path, np.array(results_mat))

            snapshots_mat = list(np.load(snapshots_file_path))
            snapshots_mat.append(snapshot_final)
            np.save(snapshots_file_path, np.array(snapshots_mat))

            reactions_mat = list(np.load(reactions_file_path))
            reactions_mat.append(reactions)
            np.save(reactions_file_path, np.array(reactions_mat))

            #q=optim_result.x
            # q0=q

        