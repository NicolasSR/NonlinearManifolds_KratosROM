import numpy as np
import pandas as pd
import tensorflow as tf
import abc

import matplotlib.pyplot as plt


class Base_POD_PrePostProcessor(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def configure_processor(self, S):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def preprocess_input_data(self, snapshot):
        """ Define in subclass"""

    @abc.abstractmethod
    def preprocess_input_data_tf(self, snapshot_tensor):
        """ Define in subclass"""

    @abc.abstractmethod
    def postprocess_output_data(self, pred_snapshot, _):
        """ Define in subclass"""

    @abc.abstractmethod
    def postprocess_output_data_tf(self, pred_snapshot_tensor, _):
        """ Define in subclass"""

class Identity_POD_PrePostProcessor(Base_POD_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi=None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, q_size):
        print('No processing applied to input and output.')

        try:
            self.phi=np.load(self.dataset_path+'POD/phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing new one")
            self.phi, _, _ = np.linalg.svd(S.T)
            np.save(self.dataset_path+'POD/phi.npy', self.phi)

        self.phi = self.phi[:,:q_size]

        S_input = self.preprocess_input_data(S)
        S_recons = np.matmul(self.phi,np.matmul(self.phi.T,S_input.T)).T
        S_recons_denorm = self.postprocess_output_data(S_recons, None)
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons_denorm-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons_denorm, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])

    def get_phi(self):
        return self.phi

    def preprocess_input_data(self, snapshot):
        return snapshot
    
    def preprocess_input_data_tf(self,snapshot_tensor):
        return snapshot_tensor

    def postprocess_output_data(self, pred_snapshot,_):
        return pred_snapshot
    
    def postprocess_output_data_tf(self,pred_snapshot_tensor,_):
        return pred_snapshot_tensor