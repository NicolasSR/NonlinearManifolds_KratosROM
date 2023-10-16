import numpy as np
import pandas as pd
import tensorflow as tf
import abc

import matplotlib.pyplot as plt


class Base_Quad_PrePostProcessor(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def configure_processor(self, S, crop_mat_tf, crop_mat_scp):
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

class ScaleGlobal_TF_Quad_PrePostProcessor(Base_Quad_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.scale_factor = None
        self.scale_factor_tf = None
        self.crop_mat_tf = None
        self.crop_mat_scp = None
        self.phi_cropped=None
        self.H_scaled_cropped=None
        self.dataset_path=working_path+dataset_path

    def get_quadratic_q(self, x_true, phi):
        q_mat=np.matmul(phi.T,x_true.T).T
        print('q shape inside:', q_mat.shape)

        ones = np.ones((q_mat.shape[1],q_mat.shape[1]))
        mask = np.array(np.triu(ones,0), dtype=bool, copy=False)

        q_quad_mat=[]
        for q in q_mat:
            out_pr = np.matmul(np.expand_dims(q,axis=1), np.expand_dims(q,axis=0))
            q_quad_mat.append(out_pr[mask])
        q_quad_mat=np.array(q_quad_mat, copy=False)
        return q_quad_mat, q_mat

    def configure_processor(self, S, crop_mat_tf, crop_mat_scp, q_size):
        print('Applying scaling of S and cropping of fixed DoFs')

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        try:
            phi=np.load(self.dataset_path+'POD/phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Please generate one through training a POD model")
            exit()

        phi=phi[:,:q_size]
        q = np.matmul(phi.T,S.T).T

        self.scale_factor=1.0/np.max(abs(q))
        self.scale_factor_tf=tf.constant(self.scale_factor)

        S_scaled=S*self.scale_factor
        q_quad_scaled, q_scaled =self.get_quadratic_q(S_scaled, phi)

        try:
            H_scaled=np.load(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'_scaled'+str(self.scale_factor))
        except IOError:
            print("No precomputed H_reduced matrix found for the specified q_size and scale factor. Computing a new one")
            # Pseudo inverse of Q_
            q_quad_scaled_inv = np.linalg.pinv(q_quad_scaled.T)
            # Getting H
            H_scaled = (S_scaled.T - phi@q_scaled.T)@q_quad_scaled_inv
            print(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'_scaled'+str(self.scale_factor))
            np.save(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'_scaled'+str(self.scale_factor),H_scaled)


        self.phi_cropped = self.crop_mat_scp.T@phi
        self.H_scaled_cropped = self.crop_mat_scp.T@H_scaled

        S_input = self.preprocess_input_data(S)
        S_recons_lin = np.matmul(self.phi_cropped,np.matmul(self.phi_cropped.T,S_input.T))
        S_recons_quad = np.matmul(self.H_scaled_cropped,q_quad_scaled.T)
        S_recons_quad_proj= np.matmul(self.phi_cropped,np.matmul(self.phi_cropped.T,S_recons_quad))
        S_recons = (S_recons_lin + S_recons_quad - S_recons_quad_proj).T
        S_recons_denorm = self.postprocess_output_data(S_recons, None)
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons_denorm-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons_denorm, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])

    def get_phi_and_H(self):
        return self.phi_cropped, self.H_scaled_cropped

    def preprocess_input_data(self, snapshot):
        output_data=snapshot.copy()
        print(self.crop_mat_scp.shape)
        output_data=(self.crop_mat_scp.T@output_data.T).T
        output_data*=self.scale_factor
        return output_data
    
    def preprocess_input_data_tf(self,snapshot_tensor):
        output_tensor = tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, snapshot_tensor, adjoint_a=True, adjoint_b=True, name=None))
        output_tensor = output_tensor*self.scale_factor_tf
        return output_tensor

    def postprocess_output_data(self, pred_snapshot,_):
        output_data=pred_snapshot.copy()
        output_data/=self.scale_factor
        output_data=(self.crop_mat_scp@output_data.T).T
        return output_data
    
    def postprocess_output_data_tf(self,pred_snapshot_tensor,_):
        output_tensor=pred_snapshot_tensor*(1.0/self.scale_factor)
        output_tensor = tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True, name=None))
        return output_tensor
    

class Identity_Quad_PrePostProcessor(Base_Quad_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi=None
        self.H=None
        self.dataset_path=working_path+dataset_path

    def get_quadratic_q(self, x_true):
        q_mat=np.matmul(self.phi.T,x_true.T).T

        ones = np.ones((q_mat.shape[1],q_mat.shape[1]))
        mask = np.array(np.triu(ones,0), dtype=bool, copy=False)

        q_quad_mat=[]
        for q in q_mat:
            out_pr = np.matmul(np.expand_dims(q,axis=1), np.expand_dims(q,axis=0))
            q_quad_mat.append(out_pr[mask])
        q_quad_mat=np.array(q_quad_mat, copy=False)
        return q_quad_mat, q_mat

    def configure_processor(self, S, crop_mat_tf, crop_mat_scp, q_size):
        print('No processing applied to input and output.')

        try:
            self.phi=np.load(self.dataset_path+'POD/phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Please generate one through training a POD model")
            exit()

        self.phi = self.phi[:,:q_size]

        q_quad, q = self.get_quadratic_q(S)
        print('Shape q: ', q.shape)
        print('Shape q_quad: ', q_quad.shape)

        try:
            self.H_mat=np.load(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'.npy')
        except IOError:
            print("No precomputed H_reduced matrix found for the specified q_size and scale factor. Computing a new one")
            # Pseudo inverse of Q_
            q_quad_scaled_inv = np.linalg.pinv(q_quad.T)
            # Getting H
            self.H_mat = (S.T - self.phi@q.T)@q_quad_scaled_inv
            print(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'.npy')
            np.save(self.dataset_path+'Quad/H_matrix_n'+str(q_size)+'.npy',self.H_mat)

        S_input = self.preprocess_input_data(S)
        S_recons_lin = np.matmul(self.phi,np.matmul(self.phi.T,S_input.T))
        S_recons_quad = np.matmul(self.H_mat,q_quad.T)
        S_recons_quad_proj= np.matmul(self.phi,np.matmul(self.phi.T,S_recons_quad))
        S_recons = (S_recons_lin + S_recons_quad - S_recons_quad_proj).T
        S_recons_denorm = self.postprocess_output_data(S_recons, None)
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons_denorm-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons_denorm, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])

    def get_phi_and_H(self):
        return self.phi, self.H_mat

    def preprocess_input_data(self, snapshot):
        return snapshot
    
    def preprocess_input_data_tf(self,snapshot_tensor):
        return snapshot_tensor

    def postprocess_output_data(self, pred_snapshot,_):
        return pred_snapshot
    
    def postprocess_output_data_tf(self,pred_snapshot_tensor,_):
        return pred_snapshot_tensor