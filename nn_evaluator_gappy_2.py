import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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

from Utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, mean_l2_error, forbenius_error, l2_error_list
from Utils.matrix_to_gid_output import output_GID_from_matrix

tf.keras.backend.set_floatx('float64')

class NN_Evaluator():

    def __init__(self, working_path, model_path, GID_FOM_filename, best, test_validation=False, test_small=False):
        self.working_path=working_path
        self.model_path=working_path+model_path
        self.results_path=self.model_path+'reconstruction_evaluation_results/'
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

        self.GID_FOM_filename = GID_FOM_filename

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        with open(self.model_path+"train_config.npy", "rb") as train_config_file:
            self.train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        print(self.train_config)
        self.dataset_path=working_path+self.train_config['dataset_path']

        self.test_validation=test_validation
        self.test_small=test_small
        if self.test_validation:
            self.name_complement='_test_validation_'
        elif self.test_small:
            self.name_complement='_test_small_'
        else:
            self.name_complement = ''


    def get_last_best_filename(self, model_weights_path, prefix):
        if os.path.exists(prefix+'best.npy'):
            highest_filename=prefix+'best.npy'
        else:
            matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
            highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def prepare_evaluation_data(self):

        if self.test_validation==False and self.test_small==False:
            # S_test=np.load(self.dataset_path+'S_large_dataset.npy').T
            S_test=np.load(self.dataset_path+'S_test.npy')
            R_test=np.load(self.dataset_path+'R_test.npy')
            # F_test=np.load(self.dataset_path+'F_large_dataset.npy')
            if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
                F_test=np.load(self.dataset_path+'F_test.npy')
            elif isinstance(self.kratos_simulation, FluidDynamics_KratosSimulator):
                F_test=None
            else:
                 F_test=None

        elif self.test_validation:
            S_test=np.load(self.dataset_path+'S_val.npy')
            R_test=np.load(self.dataset_path+'R_val.npy')
            if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
                F_test=np.load(self.dataset_path+'F_val.npy')
            elif isinstance(self.kratos_simulation, FluidDynamics_KratosSimulator):
                F_test=None
            else:
                 F_test=None

        elif self.test_small:
            S_test=np.load(self.dataset_path+'S_test_small.npy')
            R_test=np.load(self.dataset_path+'R_test_small.npy')
            if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
                F_test=np.load(self.dataset_path+'F_test_small.npy')
            elif isinstance(self.kratos_simulation, FluidDynamics_KratosSimulator):
                F_test=None
            else:
                 F_test=None

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
        if sim_type == 'structural':
            return StructuralMechanics_KratosSimulator
        elif sim_type == 'fluid':
            return FluidDynamics_KratosSimulator
        else:
            print('Invalid Kratos Analysis type')
            return None        
    
    def get_pred_snapshots_matrix(self, x_true):
        file_name = 's_pred_matrix'+self.best_name_part+self.name_complement+'.npy'
        try:
            x_pred = np.load(self.results_path+file_name)
        except IOError:
            print("No precomputed S_pred matrix. Computing new one")
            x_true_norm, aux_norm_data = self.prepost_processor.preprocess_input_data(x_true)
            x_pred_norm = self.network(x_true_norm).numpy()
            x_pred = self.prepost_processor.postprocess_output_data(x_pred_norm, (x_true_norm,aux_norm_data))
            np.save(self.results_path+file_name, x_pred)
        return x_pred
    
    def get_pred_r_noForce_matrix(self, x_pred):
        file_name = 'r_noForce_pred_matrix'+self.best_name_part+self.name_complement+'.npy'
        try:
            r_noForce_pred = np.load(self.results_path+file_name)
        except IOError:
            print("No precomputed R_noForce_pred matrix. Computing new one")
            r_noForce_pred = self.kratos_simulation.get_r_array(x_pred)
            np.save(self.results_path+file_name, r_noForce_pred)
        return r_noForce_pred
    
    def get_pred_r_force_matrix(self, x_pred, F_test):
        file_name = 'r_force_pred_matrix'+self.best_name_part+self.name_complement+'.npy'
        try:
            r_force_pred = np.load(self.results_path+file_name)
        except IOError:
            print("No precomputed R_force_pred matrix. Computing new one")
            r_force_pred = self.kratos_simulation.get_r_forces_array(x_pred, F_test)
            np.save(self.results_path+file_name, r_force_pred)
        return r_force_pred
    
    def display_failure_region(self, R_test, R_noForce_pred, tolerance, F):
        g_list = l2_error_list(R_test, R_noForce_pred) - tolerance
        sc=plt.scatter(F[:,0,0],F[:,20,1], c=g_list, s=4, cmap='jet')
        # sc=plt.scatter(F[:,0,0],F[:,20,1], c=g_list, s=4, cmap='jet', norm=matplotlib.colors.LogNorm())
        plt.colorbar(sc)
        plt.show()
        F_parsed=np.concatenate((np.expand_dims(F[:,0,0],axis=1),np.expand_dims(F[:,20,1],axis=1)), axis=1)
        return g_list
    
    def SAIS_recursion(self, F, g_list):
        F=np.concatenate((np.expand_dims(F[:,0,0],axis=1),np.expand_dims(F[:,20,1],axis=1)), axis=1)
        sorted_indices = g_list.argsort()[::-1]
        # print(sorted_indices)
        # print(g_list[sorted_indices])
        F_sorted = F[sorted_indices]
        N_eta=int(np.where(g_list>0)[0].shape[0])
        N_p=int(0.2*F.shape[0])
        print('N_eta: ', N_eta, '   N_p: ', N_p)

        D_np=F_sorted[:N_p,:]
        mu_opt=np.mean(D_np, axis=0)
        covar_mat=np.cov(D_np.T)
        print('Mean: ', mu_opt)
        print('Covariance: ', covar_mat)

        # candidate_params = np.random.multivariate_normal(mu_opt, covar_mat, size=1000)
        # plt.scatter(candidate_params[:,0],candidate_params[:,1])
        # plt.show()
            
    
    def display_snapshot_relative_errors(self, x_true, x_pred):
        l2_error=mean_relative_l2_error(x_true,x_pred)
        forb_error=relative_forbenius_error(x_true,x_pred)
        print('Snapshot Mean rel L2 error: ', l2_error)
        print('Snapshot Rel Forb. error: ', forb_error)

    def display_r_diff_relative_errors(self, r_noForce_true, r_noForce_pred):
        l2_error=mean_relative_l2_error(r_noForce_true,r_noForce_pred)
        forb_error=relative_forbenius_error(r_noForce_true,r_noForce_pred)
        print('Residual NoForce Mean rel L2 error l:', l2_error)
        print('Residual NoForce Rel Forb. error l:', forb_error)

    def display_r_norm_errors(self, r_force_pred):
        l2_error=mean_l2_error(r_force_pred, 0.0)
        forb_error=forbenius_error(r_force_pred, 0.0)
        print('Residual Force Mean rel L2 error l:', l2_error)
        print('Residual Force Rel Forb. error l:', forb_error)

    def get_GID_FOM_reconstruction(self):

        output_filename=self.results_path+'recons_'+self.GID_FOM_filename[:-5]+self.best_name_part+self.name_complement

        if os.path.isfile(output_filename+'.post.res'):
            print('GID file for the reconstruction already exists')
        else:
            print('Generating GID file for the reconstruction')
            recons_project_parameters_filename = self.train_config["dataset_path"]+'ProjectParameters_recons.json'
            with open(recons_project_parameters_filename, "r") as recons_project_parameters_file:
                recons_project_parameters=json.load(recons_project_parameters_file)
            mdpa_filename=recons_project_parameters["solver_settings"]["model_import_settings"]["input_filename"]

            x_FOM = np.load(self.dataset_path+'FOM/'+self.GID_FOM_filename)
            F_FOM = np.load(self.dataset_path+'FOM/'+'POINTLOADS'+self.GID_FOM_filename[3:])

            x_true_norm, aux_norm_data = self.prepost_processor.preprocess_input_data(x_FOM)
            x_pred_norm = self.network(x_true_norm).numpy()
            x_pred = self.prepost_processor.postprocess_output_data(x_pred_norm, (x_true_norm,aux_norm_data))
            np.save(output_filename+'.npy', x_pred)
            reactions = self.kratos_simulation.get_r_forces_withDirich_array(x_pred, F_FOM)

            output_GID_from_matrix(mdpa_filename, output_filename, x_pred, reactions)

    def execute_evaluation(self):

        # Select the architecture to use
        arch_factory = self.architecture_factory_selector(self.train_config["architecture"])

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation_class=self.kratos_simulator_selector(self.train_config["sim_type"])
        self.kratos_simulation = kratos_simulation_class(self.working_path, self.train_config)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        # Select the type of preprocessing (normalisation)
        self.prepost_processor=arch_factory.prepost_processor_selector(self.working_path, self.train_config["dataset_path"])

        F_train = np.load(self.train_config['dataset_path']+'F_train.npy')
        non_cropped_zone = [[-1000,1000],[-1000,1000]]

        gap_sample_ids_train=np.squeeze(np.argwhere((F_train[:,0,0]>non_cropped_zone[0,0]) & (F_train[:,0,0]<non_cropped_zone[0,1]) & (F_train[:,6,1]>non_cropped_zone[1,0]) & (F_train[:,6,1]<non_cropped_zone[1,1])),axis=1)

        S_FOM_orig = arch_factory.get_orig_fom_snapshots(self.train_config['dataset_path'])
        S_FOM_orig = S_FOM_orig[gap_sample_ids_train]

        S_FOM_orig = arch_factory.get_orig_fom_snapshots(self.train_config['dataset_path'])
        arch_factory.configure_processor_non_saved(self.prepost_processor, S_FOM_orig, crop_mat_tf, crop_mat_scp)

        print('======= Instantiating TF Model =======')
        self.network, _, __ = arch_factory.define_network(self.prepost_processor, self.kratos_simulation)
        
        print('======= Loading TF Model weights =======')
        self.network.load_weights(self.model_weights_path+self.model_weights_filename)

        S_test, R_test, F_test = self.prepare_evaluation_data()
        print('Shape S_test: ', S_test.shape)
        print('Shape R_test: ', R_test.shape)
        if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
            print('Shape F_test: ', F_test.shape)

        gap_sample_ids_test=np.squeeze(np.argwhere((F_test[:,0,0]>non_cropped_zone[0,0]) & (F_test[:,0,0]<non_cropped_zone[0,1]) & (F_test[:,6,1]>non_cropped_zone[1,0]) & (F_test[:,6,1]<non_cropped_zone[1,1])),axis=1)

        S_test = np.delete(S_test, gap_sample_ids_test, axis=0)
        R_test = np.delete(R_test, gap_sample_ids_test, axis=0)
        F_test = np.delete(F_test, gap_sample_ids_test, axis=0)

        S_pred = self.get_pred_snapshots_matrix(S_test)
        R_noForce_pred = self.get_pred_r_noForce_matrix(S_pred)
        if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
            R_force_pred = self.get_pred_r_force_matrix(S_pred, F_test)

        # g_list = self.display_failure_region(0, R_force_pred, 1000, F_test)
        # sorted_indices = g_list.argsort()[::-1]
        # F_test_sorted = F_test[sorted_indices]
        # S_test_sorted = S_test[sorted_indices]

        # F_extra_samples = F_test_sorted[:500]
        # S_extra_samples = S_test_sorted[:500]
        # R_noForce_pred = self.get_pred_r_noForce_matrix(S_extra_samples)

        # print('R: ', R_noForce_pred.shape)
        # print(R_noForce_pred)
        # print('S: ', S_extra_samples.shape)
        # print(S_extra_samples)
        # print('F: ', F_extra_samples.shape)
        # print(F_extra_samples)

        # np.save('S_new_extra_samples.npy', S_extra_samples)
        # np.save('F_new_extra_samples.npy', F_extra_samples)


        # self.SAIS_recursion(F_test, g_list)

        self.display_snapshot_relative_errors(S_test, S_pred)
        self.display_r_diff_relative_errors(R_test, R_noForce_pred)
        if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
            self.display_r_norm_errors(R_force_pred)

        if isinstance(self.kratos_simulation, StructuralMechanics_KratosSimulator):
            self.get_GID_FOM_reconstruction()