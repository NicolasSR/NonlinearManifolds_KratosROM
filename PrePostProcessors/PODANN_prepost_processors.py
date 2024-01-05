import numpy as np
import tensorflow as tf
import abc

from PrePostProcessors.base_prepost_processor import Base_PrePostProcessor

# class Base_PODANN_PrePostProcessor(abc.ABC):

#     def __init__(self):
#         super().__init__()

#     @abc.abstractmethod
#     def configure_processor(self, S, crop_mat_tf, crop_mat_scp):
#         """ Define in subclass"""

#     @abc.abstractmethod
#     def preprocess_nn_output_data(self, snapshot):
#         """ Define in subclass"""

#     @abc.abstractmethod
#     def preprocess_nn_output_data_tf(self, snapshot_tensor):
#         """ Define in subclass"""
    
#     @abc.abstractmethod
#     def preprocess_input_data(self, snapshot):
#         """ Define in subclass"""

#     @abc.abstractmethod
#     def preprocess_input_data_tf(self, snapshot_tensor):
#         """ Define in subclass"""

#     @abc.abstractmethod
#     def postprocess_output_data(self, q_sup, q_inf):
#         """ Define in subclass"""

#     @abc.abstractmethod
#     def postprocess_output_data_tf(self, q_sup_tensor, q_inf_tensor):
#         """ Define in subclass"""


class SVD_White_NoStand_PODANN_PrePostProcessor(Base_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sup = None
        self.phi_sup_tf = None
        self.sigma_inf = None
        self.sigma_inf_tf = None
        self.sigma_sup = None
        self.sigma_sup_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization for PODANN architecture')

        try:
            self.phi=np.load(self.dataset_path+'PODANN/phi_whitenostand.npy')
            self.sigma=np.load(self.dataset_path+'PODANN/sigma_whitenostand.npy')
        except IOError:
            print("No precomputed phi_whitenostand or sigma_whitenostand matrix found. Computing a new set")
            S_scaled=S/np.sqrt(S.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(S_scaled.T)
            np.save(self.dataset_path+'PODANN/phi_whitenostand.npy', self.phi)
            np.save(self.dataset_path+'PODANN/sigma_whitenostand.npy', self.sigma)

        self.phi_inf=self.phi[:,:svd_inf_size].copy()
        self.sigma_inf=self.sigma[:svd_inf_size].copy()
        self.phi_sup=self.phi[:,svd_inf_size:svd_sup_size].copy()
        self.sigma_sup=self.sigma[svd_inf_size:svd_sup_size].copy()
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Sigma_inf array shape: ', self.sigma_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sup.shape)
        print('Sigma_sgs array shape: ', self.sigma_sup.shape)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.sigma_inf_tf=tf.constant(self.sigma_inf)
        self.phi_sup_tf=tf.constant(self.phi_sup)
        self.sigma_sup_tf=tf.constant(self.sigma_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S)
        S_recons_aux2, _ =self.preprocess_input_data(S)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, None))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S.shape[0]))

    def configure_processor_non_saved(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization for PODANN architecture. Forced to calculate projection operators.')

        S_scaled=S/np.sqrt(S.shape[0])
        self.phi,self.sigma, _ = np.linalg.svd(S_scaled.T)

        self.phi_inf=self.phi[:,:svd_inf_size].copy()
        self.sigma_inf=self.sigma[:svd_inf_size].copy()
        self.phi_sup=self.phi[:,svd_inf_size:svd_sup_size].copy()
        self.sigma_sup=self.sigma[svd_inf_size:svd_sup_size].copy()
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Sigma_inf array shape: ', self.sigma_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sup.shape)
        print('Sigma_sgs array shape: ', self.sigma_sup.shape)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.sigma_inf_tf=tf.constant(self.sigma_inf)
        self.phi_sup_tf=tf.constant(self.phi_sup)
        self.sigma_sup_tf=tf.constant(self.sigma_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S)
        S_recons_aux2, _ =self.preprocess_input_data(S)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, None))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S.shape[0]))

    def preprocess_nn_output_data(self, snapshot):
        # Returns q_sup from input snapshots
        output_data=snapshot.copy()
        output_data=np.divide(np.matmul(self.phi_sup.T,output_data.T).T,self.sigma_sup)
        # output_data=np.matmul(self.phi_sup.T,output_data.T).T
        return output_data
    
    def preprocess_nn_output_data_tf(self,snapshot_tensor):
        # Returns q_sup from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,snapshot_tensor,transpose_a=True,transpose_b=True))/self.sigma_sup_tf
        return output_tensor
    
    def preprocess_input_data(self, snapshot):
        # Returns q_inf from input snapshots
        output_data=snapshot.copy()
        output_data=np.divide(np.matmul(self.phi_inf.T,output_data.T).T,self.sigma_inf)
        # output_data=np.matmul(self.phi_inf.T,output_data.T).T
        return output_data, None
    
    def preprocess_input_data_tf(self, snapshot_tensor):
        # Returns q_inf from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,snapshot_tensor,transpose_a=True,transpose_b=True))/self.sigma_inf_tf
        return output_tensor, None

    def postprocess_output_data(self, q_sup, aux_norm_data):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf, _ = aux_norm_data
        output_data_1=q_inf.copy()
        output_data_1=np.matmul(self.phi_inf,np.multiply(output_data_1, self.sigma_inf).T).T
        output_data_2=q_sup.copy()
        output_data_2=np.matmul(self.phi_sup,np.multiply(output_data_2, self.sigma_sup).T).T
        output_data = output_data_1 + output_data_2
        return output_data
    
    def postprocess_output_data_tf(self, q_sup_tensor, aux_norm_tensors):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf_tensor, _ = aux_norm_tensors
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,q_inf_tensor*self.sigma_inf_tf,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,q_sup_tensor*self.sigma_sup_tf,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2
        return output_tensor
    
    def get_phi_matrices(self):
        return self.phi_inf, self.phi_sup
    
    def get_sigma_vectors(self):
        return self.sigma_inf, self.sigma_sup
    

class SVD_White_NoStand_Cropping_PODANN_PrePostProcessor(Base_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.crop_mat_scp=None
        self.crop_mat_tf=None
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sup = None
        self.phi_sup_tf = None
        self.sigma_inf = None
        self.sigma_inf_tf = None
        self.sigma_sup = None
        self.sigma_sup_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization for PODANN architecture')

        self.crop_mat_scp=crop_mat_scp
        self.crop_mat_tf=crop_mat_tf

        S_bound=(self.crop_mat_scp@S.T).T
        S_comp=S-S_bound

        try:
            self.phi=np.load(self.dataset_path+'PODANN/phi_whitenostand_crop.npy')
            self.sigma=np.load(self.dataset_path+'PODANN/sigma_whitenostand_crop.npy')
        except IOError:
            print("No precomputed phi_whitenostand_crop or sigma_whitenostand_crop matrix found. Computing a new set")
            S_scaled=S_comp/np.sqrt(S.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(S_scaled.T)
            np.save(self.dataset_path+'PODANN/phi_whitenostand_crop.npy', self.phi)
            np.save(self.dataset_path+'PODANN/sigma_whitenostand_crop.npy', self.sigma)

        self.phi_inf=self.phi[:,:svd_inf_size].copy()
        self.sigma_inf=self.sigma[:svd_inf_size].copy()
        self.phi_sup=self.phi[:,svd_inf_size:svd_sup_size].copy()
        self.sigma_sup=self.sigma[svd_inf_size:svd_sup_size].copy()
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Sigma_inf array shape: ', self.sigma_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sup.shape)
        print('Sigma_sgs array shape: ', self.sigma_sup.shape)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.sigma_inf_tf=tf.constant(self.sigma_inf)
        self.phi_sup_tf=tf.constant(self.phi_sup)
        self.sigma_sup_tf=tf.constant(self.sigma_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S)
        S_recons_aux2, S_recons_bound=self.preprocess_input_data(S)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, S_recons_bound))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S.shape[0]))

    def preprocess_nn_output_data(self, snapshot):
        # Returns q_sup from input snapshots
        total_snapshot=snapshot.copy().T
        snapshot_bound=self.crop_mat_scp@total_snapshot
        output_data=total_snapshot-snapshot_bound
        output_data=np.divide(np.matmul(self.phi_sup.T,output_data).T,self.sigma_sup)
        return output_data
    
    def preprocess_nn_output_data_tf(self,snapshot_tensor):
        # Returns q_sup from input snapshots
        snapshot_bound=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, snapshot_tensor, adjoint_a=False, adjoint_b=True, name=None))
        output_tensor=snapshot_tensor-snapshot_bound
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,output_tensor,transpose_a=True,transpose_b=True))/self.sigma_sup_tf
        return output_tensor
    
    def preprocess_input_data(self, snapshot):
        # Returns q_inf from input snapshots
        total_snapshot=snapshot.copy().T
        snapshot_bound=self.crop_mat_scp@total_snapshot
        output_data=total_snapshot-snapshot_bound
        output_data=np.divide(np.matmul(self.phi_inf.T,output_data).T,self.sigma_inf)
        return output_data, snapshot_bound.T
    
    def preprocess_input_data_tf(self, snapshot_tensor):
        # Returns q_inf from input snapshots
        snapshot_bound=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, snapshot_tensor, adjoint_a=False, adjoint_b=True, name=None))
        output_tensor=snapshot_tensor-snapshot_bound
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,snapshot_tensor,transpose_a=True,transpose_b=True))/self.sigma_inf_tf
        return output_tensor, snapshot_bound

    def postprocess_output_data(self, q_sup, aux_data):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf, snapshot_bound = aux_data
        output_data_1=q_inf.copy()
        output_data_1=np.matmul(self.phi_inf,np.multiply(output_data_1, self.sigma_inf).T).T
        output_data_2=q_sup.copy()
        output_data_2=np.matmul(self.phi_sup,np.multiply(output_data_2, self.sigma_sup).T).T
        output_data = output_data_1 + output_data_2
        output_data += snapshot_bound
        return output_data
    
    def postprocess_output_data_tf(self, q_sup_tensor, aux_tensors):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf_tensor, snapshot_bound = aux_tensors
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,q_inf_tensor*self.sigma_inf_tf,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,q_sup_tensor*self.sigma_sup_tf,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2
        output_tensor += snapshot_bound
        return output_tensor
    
    def get_training_data(self, arch_config):
            
        input_data, target_data, val_input, val_target = super().get_training_data(arch_config)
            
        _, train_snapshot_bound = self.preprocess_input_data(target_data[0])
        _, val_snapshot_bound = self.preprocess_input_data(val_target[0])

        target_data=(target_data[0], target_data[1], train_snapshot_bound)
        val_target=(val_target[0], val_target[1], val_snapshot_bound)

        return input_data, target_data, val_input, val_target
    


class SVD_Rerange_PODANN_PrePostProcessor(Base_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sup = None
        self.phi_sup_tf = None
        self.rerange_scale = None
        self.rerange_scale_tf = None
        self.rerange_offset = None
        self.rerange_offset_tf = None
        self.rerange_scale_sup = None
        self.rerange_scale_sup_tf = None
        self.rerange_offset_sup = None
        self.rerange_offset_sup_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization for PODANN architecture')

        try:
            self.phi=np.load(self.dataset_path+'POD/phi.npy')
        except IOError:
            print("No precomputed phi. Train a POD model to obtain it")
            exit()

        self.phi_inf=self.phi[:,:svd_inf_size].copy()
        self.phi_sup=self.phi[:,svd_inf_size:svd_sup_size].copy()
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sup.shape)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.phi_sup_tf=tf.constant(self.phi_sup)

        Q_inf=self.phi_inf.T@S.T
        Q_inf=Q_inf.T
        self.rerange_scale=np.max(Q_inf,axis=0)-np.min(Q_inf,axis=0)
        self.rerange_offset=np.min(Q_inf,axis=0)
        self.rerange_scale_tf=tf.constant(self.rerange_scale)
        self.rerange_offset_tf=tf.constant(self.rerange_offset)

        Q_sup=self.phi_sup.T@S.T
        Q_sup=Q_sup.T
        self.rerange_scale_sup=np.max(Q_sup,axis=0)-np.min(Q_sup,axis=0)
        self.rerange_offset_sup=np.min(Q_sup,axis=0)
        self.rerange_scale_sup_tf=tf.constant(self.rerange_scale_sup)
        self.rerange_offset_sup_tf=tf.constant(self.rerange_offset_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S)
        S_recons_aux2,_=self.preprocess_input_data(S)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, None))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S.shape[0]))

    def preprocess_nn_output_data(self, snapshot):
        # Returns q_sup from input snapshots
        output_data=snapshot.copy()
        output_data=np.divide(np.matmul(self.phi_sup.T,output_data.T).T-self.rerange_offset_sup, self.rerange_scale_sup)
        # output_data=np.matmul(self.phi_sup.T,output_data.T).T
        return output_data
    
    def preprocess_nn_output_data_tf(self,snapshot_tensor):
        # Returns q_sup from input snapshots
        output_tensor=(tf.transpose(tf.linalg.matmul(self.phi_sup_tf,snapshot_tensor,transpose_a=True,transpose_b=True))-self.rerange_offset_sup_tf)/self.rerange_scale_sup_tf
        return output_tensor
    
    def preprocess_input_data(self, snapshot):
        # Returns q_inf from input snapshots
        output_data=snapshot.copy()
        print(output_data.shape)
        output_data=np.divide(np.matmul(self.phi_inf.T,output_data.T).T-self.rerange_offset, self.rerange_scale)
        print(output_data.shape)
        return output_data, None
    
    def preprocess_input_data_tf(self, snapshot_tensor):
        # Returns q_inf from input snapshots
        print(snapshot_tensor.shape)
        output_tensor=(tf.transpose(tf.linalg.matmul(self.phi_inf_tf,snapshot_tensor,transpose_a=True,transpose_b=True))-self.rerange_offset_tf)/self.rerange_scale_tf
        print(output_tensor.shape)
        return output_tensor, None

    def postprocess_output_data(self, q_sup, aux_norm_data):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf, _ = aux_norm_data
        # output_data_1=np.multiply(q_inf.copy(),self.rerange_scale)+self.rerange_offset
        # output_data_1=np.matmul(self.phi_inf,output_data_1.T).T
        # output_data_2=q_sup.copy()
        # output_data_2=np.matmul(self.phi_sup,output_data_2.T).T
        # output_data = output_data_1 + output_data_2

        q_inf_tensor=tf.constant(q_inf)
        q_sup_tensor=tf.constant(q_sup)
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,(q_inf_tensor*self.rerange_scale_tf)+self.rerange_offset_tf,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,q_sup_tensor,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2
        output_data=output_tensor.numpy()
        return output_data
    
    def postprocess_output_data_tf(self, q_sup_tensor, aux_norm_tensors):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf_tensor, _ = aux_norm_tensors
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,(q_inf_tensor*self.rerange_scale_tf)+self.rerange_offset_tf,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,(q_sup_tensor*self.rerange_scale_sup_tf)+self.rerange_offset_sup_tf,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2
        return output_tensor
    
    def get_phi_matrices(self):
        return self.phi_inf, self.phi_sup
    


class SVD_PODANN_PrePostProcessor(Base_PrePostProcessor):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sup = None
        self.phi_sup_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp):
        print('Applying SVD for PODANN architecture')

        try:
            self.phi=np.load(self.dataset_path+'POD/phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Please train a POD case to generate it")

        self.phi_inf=self.phi[:,:svd_inf_size].copy()
        self.phi_sup=self.phi[:,svd_inf_size:svd_sup_size].copy()
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sup.shape)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.phi_sup_tf=tf.constant(self.phi_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S)
        S_recons_aux2, _ =self.preprocess_input_data(S)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, None))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S.shape[0]))

    def preprocess_nn_output_data(self, snapshot):
        # Returns q_sup from input snapshots
        output_data=snapshot.copy()
        output_data=np.matmul(self.phi_sup.T,output_data.T).T
        return output_data
    
    def preprocess_nn_output_data_tf(self,snapshot_tensor):
        # Returns q_sup from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,snapshot_tensor,transpose_a=True,transpose_b=True))
        return output_tensor
    
    def preprocess_input_data(self, snapshot):
        # Returns q_inf from input snapshots
        output_data=snapshot.copy()
        output_data=np.matmul(self.phi_inf.T,output_data.T).T
        return output_data, None
    
    def preprocess_input_data_tf(self, snapshot_tensor):
        # Returns q_inf from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,snapshot_tensor,transpose_a=True,transpose_b=True))
        return output_tensor, None

    def postprocess_output_data(self, q_sup, aux_norm_data):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf, _ = aux_norm_data
        output_data_1=q_inf.copy()
        output_data_1=np.matmul(self.phi_inf,output_data_1.T).T
        output_data_2=q_sup.copy()
        output_data_2=np.matmul(self.phi_sup,output_data_2.T).T
        output_data = output_data_1 + output_data_2
        return output_data
    
    def postprocess_output_data_tf(self, q_sup_tensor, aux_norm_tensors):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf_tensor, _ = aux_norm_tensors
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,q_inf_tensor,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phi_sup_tf,q_sup_tensor,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2
        return output_tensor
    
    def get_phi_matrices(self):
        return self.phi_inf, self.phi_sup
    