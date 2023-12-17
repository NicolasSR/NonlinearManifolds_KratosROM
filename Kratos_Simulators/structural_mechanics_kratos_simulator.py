import numpy as np
import scipy

import tensorflow as tf

import KratosMultiphysics as KMP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

import time

class StructuralMechanics_KratosSimulator():

    def __init__(self, working_path, train_config):
        if "project_parameters_file" in train_config:
            project_parameters_path=train_config["dataset_path"]+train_config["project_parameters_file"]
        else: 
            project_parameters_path='ProjectParameters_fom.json'
            print('LOADED DEFAULT PROJECT PARAMETERS FILE')
        with open(working_path+project_parameters_path, 'r') as parameter_file:
            parameters = KMP.Parameters(parameter_file.read())

        global_model = KMP.Model()
        self.fake_simulation = StructuralMechanicsAnalysis(global_model, parameters)
        self.fake_simulation.Initialize()
        self.fake_simulation.InitializeSolutionStep()

        self.space = KMP.UblasSparseSpace()
        self.strategy = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        self.buildsol = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        # self.scheme = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        self.scheme = self.fake_simulation._GetSolver()._GetScheme()
        # print(self.scheme)
        self.modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        self.var_utils = KMP.VariableUtils()


    # def get_crop_matrix(self):
    #     indices=[]
    #     col=0
    #     for i, node in enumerate(self.modelpart.Nodes):
    #         if not node.IsFixed(KMP.DISPLACEMENT_X):
    #             indices.append([2*i,col])
    #             col+=1
    #         if not node.IsFixed(KMP.DISPLACEMENT_Y):
    #             indices.append([2*i+1,col])
    #             col+=1
    #     num_cols=col
    #     num_rows=self.modelpart.NumberOfNodes()*2
    #     values=np.ones(num_cols)
    #     crop_mat_tf = tf.sparse.SparseTensor(
    #         indices=indices,
    #         values=values,
    #         dense_shape=[num_rows,num_cols])
    #     indices=np.asarray(indices)
    #     crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_cols]).tocsr()
        
    #     return crop_mat_tf, crop_mat_scp

    def get_crop_matrix(self):
        indices=[]
        for i, node in enumerate(self.modelpart.Nodes):
            if node.IsFixed(KMP.DISPLACEMENT_X):
                indices.append([2*i,2*i])
            if node.IsFixed(KMP.DISPLACEMENT_Y):
                indices.append([2*i+1,2*i+1])
        num_rows=self.modelpart.NumberOfNodes()*2
        values=np.ones(len(indices))
        crop_mat_tf = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_rows,num_rows])
        indices=np.asarray(indices)
        crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_rows]).tocsr()
        
        return crop_mat_tf, crop_mat_scp

    def get_cropped_dof_ids(self):
        indices=[]
        for i, node in enumerate(self.modelpart.Nodes):
            if node.IsFixed(KMP.DISPLACEMENT_X):
                indices.append(2*i)
            if node.IsFixed(KMP.DISPLACEMENT_Y):
                indices.append(2*i+1)
        return indices

    def project_prediction_vectorial_optim(self, y_pred):
        values = y_pred[0]
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full+=values

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT, values_full, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)

    def project_prediction_vectorial_optim_batch(self, y_pred):
        values = y_pred
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full+=values

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT, values_full, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)

    def project_prediction_vectorial_optim_forces(self, y_pred, f_vectors):
        values = y_pred[0]
        forces = f_vectors[0]
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full+=values

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT, values_full, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)
        
        conditions_array=self.modelpart.Conditions
        for i, condition in enumerate(conditions_array):
            condition.SetValue(SMA.LINE_LOAD, forces[i])

    def project_prediction_vectorial_optim_forces_batch(self, y_pred, f_vectors):
        values = y_pred
        forces = f_vectors
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full+=values

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT, values_full, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)
        
        conditions_array=self.modelpart.Conditions
        for i, condition in enumerate(conditions_array):
            condition.SetValue(SMA.LINE_LOAD, forces[i])

    def get_v_loss_rdiff_(self, y_pred, b_true):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)

        err_r=KMP.Vector(b_true[0]-b)

        v_loss_r = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_r, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_r)

        self.space.TransposeMult(A,err_r,v_loss_r)
        
        err_r=np.expand_dims(np.array(err_r, copy=False),axis=0)
        v_loss_r=np.expand_dims(np.array(v_loss_r, copy=False),axis=0)
        # The negative sign we should apply to A is compensated by the derivative of the loss
        
        return err_r, v_loss_r
    
    def get_v_loss_rdiff_batch_(self, y_pred, b_true):
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        err_r_list=[]
        v_loss_r_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            err_r=KMP.Vector(b_true[i].copy()-b)

            v_loss_r = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(v_loss_r, self.space.Size(b))
            self.space.SetToZeroVector(v_loss_r)

            self.space.TransposeMult(A,err_r,v_loss_r)
            
            err_r_list.append(np.expand_dims(np.array(err_r, copy=False),axis=0))
            v_loss_r_list.append(np.expand_dims(np.array(v_loss_r, copy=False),axis=0))
            # The negative sign we should apply to A is compensated by the derivative of the loss

        err_r_batch = np.concatenate(err_r_list, axis = 0)
        v_loss_r_batch = np.concatenate(v_loss_r_list, axis = 0)
        
        return err_r_batch, v_loss_r_batch
    
    def get_v_loss_rnorm_(self, y_pred, f_vec):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)
        foo = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(foo, self.space.Size(b))
        self.space.SetToZeroVector(foo)

        self.project_prediction_vectorial_optim_forces(y_pred, f_vec)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        self.buildsol.ApplyDirichletConditions(self.scheme, self.modelpart, A, foo, b)

        v_loss_r = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_r, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_r)

        err_r=b

        self.space.TransposeMult(A,err_r,v_loss_r)
        
        err_r=np.expand_dims(np.array(err_r, copy=False),axis=0)
        v_loss_r=-np.expand_dims(np.array(v_loss_r, copy=False),axis=0) # This negation is to make A negative.
        
        return err_r, v_loss_r
    
    def get_v_loss_rnorm_batch_(self, y_pred, f_vec):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        v_loss_r = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_r, self.space.Size(b))

        err_r_list=[]
        v_loss_r_list= []

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)
            foo = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(foo, self.space.Size(b))
            self.space.SetToZeroVector(foo)

            self.project_prediction_vectorial_optim_forces_batch(y_pred[i], f_vec[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)
            self.buildsol.ApplyDirichletConditions(self.scheme, self.modelpart, A, foo, b)

            self.space.SetToZeroVector(v_loss_r)

            err_r=b
            self.space.TransposeMult(A,err_r,v_loss_r)

            err_r_list.append(np.expand_dims(np.array(err_r, copy=True),axis=0))
            v_loss_r_list.append(-np.expand_dims(np.array(v_loss_r, copy=False),axis=0)) # This negation is to make A negative.

        err_r_batch = np.concatenate(err_r_list, axis = 0)
        v_loss_r_batch = np.concatenate(v_loss_r_list, axis = 0)
        
        return err_r_batch, v_loss_r_batch
    
    def get_v_loss_wdiff_(self, y_pred, b_true):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)

        y_pred_fixed=KMP.Vector(y_pred[0])

        v_loss_w = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_w, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_w)

        self.space.TransposeMult(A,y_pred_fixed,v_loss_w)

        v_loss_w-=b
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        v_loss_w=np.expand_dims(np.array(v_loss_w, copy=False),axis=0)
        # The negative sign we should apply to A is compensated by the derivative of the loss
        
        return b, v_loss_w
    
    def get_v_loss_welemdiff_(self, y_pred, y_true, b_true):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        v_loss_w = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_w, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_w)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)

        b=np.expand_dims(np.array(b, copy=False), axis=0)
        err_w=np.multiply(y_true, b_true)-np.multiply(y_pred, b)
        exu_vec=np.multiply(err_w,y_pred)[0].copy()
        exr_vec=np.multiply(err_w,b)

        exu_vec=KMP.Vector(exu_vec)

        self.space.TransposeMult(A,exu_vec,v_loss_w)

        v_loss_w=np.expand_dims(np.array(v_loss_w, copy=False),axis=0)-exr_vec
        

        # The negative sign we should apply to A is compensated by the derivative of the loss
        
        return err_w, v_loss_w
    
    def get_v_loss_wdiffdiff_(self, y_pred, y_true, b_true):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        v_loss_w = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_w, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_w)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)

        b=np.expand_dims(np.array(b, copy=False), axis=0)
        err_w=np.matmul(b_true-b, y_true)
        exu_vec=(err_w*y_true)[0].copy()

        exu_vec=KMP.Vector(exu_vec)

        self.space.TransposeMult(A,exu_vec,v_loss_w)

        v_loss_w=np.expand_dims(np.array(v_loss_w, copy=False),axis=0)
        
        # The negative sign we should apply to A is compensated by the derivative of the loss
        
        return err_w, v_loss_w
    
    def get_err_rdiff_(self, y_pred, b_true):
        b_pred = self.get_r_(y_pred)
        err_r = b_pred - b_true
        return err_r
    
    def get_err_rdiff_batch_(self, y_pred, b_true):
        b_pred = self.get_r_batch_(y_pred)
        err_r = b_pred - b_true
        return err_r
    
    def get_err_rnorm_(self, y_pred, f_vec):
        err_r = self.get_r_forces_(y_pred, f_vec)
        return err_r
    
    def get_err_rnorm_batch_(self, y_pred, f_vec):
        err_r = self.get_r_forces_batch_(y_pred, f_vec)
        return err_r
    
    def get_r_wdiff_(self, y_pred, b_true):
        b_pred = self.get_r_(y_pred)
        return b_pred
    
    def get_err_welemdiff_(self, y_pred, y_true, b_true):
        b_pred = self.get_r_(y_pred)
        b_pred=np.array(b_pred[0], copy=False)
        err_w=np.multiply(y_true, b_true)-np.multiply(y_pred, b_pred)
        err_w=np.expand_dims(err_w, axis=0)
        return err_w
    
    def get_err_wdiffdiff_(self, y_pred, y_true, b_true):
        b_pred = self.get_r_(y_pred)
        b_pred=np.array(b_pred[0], copy=False)
        err_w=np.matmul(b_true-b_pred, y_true)
        err_w=np.expand_dims(err_w, axis=0)
        return err_w
    
    def get_r_(self, y_pred):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        
        return b
    
    def get_r_batch_(self, y_pred):

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        b_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            b_list.append(np.expand_dims(np.array(b, copy=True),axis=0))

        b_batch = np.concatenate(b_list, axis = 0)
        
        return b_batch
    
    def get_A_e_vec_batch_(self, y_true, err):
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        err_r_list=[]
        v_loss_r_list=[]

        for i in range(y_true.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)

            self.project_prediction_vectorial_optim_batch(y_true[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            err_r=KMP.Vector(err[i].copy())

            v_loss_r = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(v_loss_r, self.space.Size(b))
            self.space.SetToZeroVector(v_loss_r)

            self.space.TransposeMult(A,err_r,v_loss_r)
            
            err_r_list.append(np.expand_dims(np.array(err_r, copy=False),axis=0))
            v_loss_r_list.append(np.expand_dims(np.array(v_loss_r, copy=False),axis=0))
            # The negative sign we should apply to A is compensated by the derivative of the loss

        err_r_batch = np.concatenate(err_r_list, axis = 0)
        v_loss_r_batch = np.concatenate(v_loss_r_list, axis = 0)
        
        return v_loss_r_batch
    
    def get_A_(self, y_pred):

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        xD  = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(xD, self.space.Size(b))

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)
        self.space.SetToZeroVector(xD)

        self.project_prediction_vectorial_optim_batch(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        # self.buildsol.ApplyDirichletConditions(self.scheme,self.modelpart,A,xD,b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))
        
        return Ascipy
    
    def get_r_forces_(self, y_pred, f_vectors):
        
        # aux = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroVector(b)
        foo = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(foo, self.space.Size(b))
        self.space.SetToZeroVector(foo)

        self.project_prediction_vectorial_optim_forces(y_pred, f_vectors)

        self.buildsol.BuildRHS(self.scheme, self.modelpart, b)
        # self.buildsol.ApplyDirichletConditions(self.scheme, self.modelpart, aux, foo, b)
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        
        return b
    
    def get_r_forces_batch_(self, y_pred, f_vectors):
        
        b = self.strategy.GetSystemVector()

        b_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroVector(b)

            self.project_prediction_vectorial_optim_forces_batch(y_pred[i], f_vectors[i])

            self.buildsol.BuildRHS(self.scheme, self.modelpart, b)
            
            b_list.append(np.expand_dims(np.array(b, copy=True),axis=0))
        
        b_batch = np.concatenate(b_list, axis = 0)

        return b_batch
    
    def get_r_forces_withDirich_(self, y_pred, f_vectors):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim_forces(y_pred, f_vectors)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        
        return b
    
    def get_dofs_with_conditions(self):
        dofs_list=set()
        conditions_array=self.modelpart.Conditions
        for condition in conditions_array:
            for node in condition.GetNodes():
                dofs_list.add(node.Id)
        dofs_list=np.array(list(dofs_list), copy=False)-1
        dofs_list=np.concatenate([dofs_list*2, dofs_list*2+1])
        return dofs_list

    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_rdiff(self, y_pred, b_true):
        y,w = tf.numpy_function(self.get_v_loss_rdiff_, [y_pred, b_true], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_rdiff_batch(self, y_pred, b_true):
        y,w = tf.numpy_function(self.get_v_loss_rdiff_batch_, [y_pred, b_true], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_rdiff(self, y_pred, b_true):
        y = tf.numpy_function(self.get_err_rdiff_, [y_pred, b_true], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_rdiff_batch(self, y_pred, b_true):
        y = tf.numpy_function(self.get_err_rdiff_batch_, [y_pred, b_true], (tf.float64))
        return y

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
    def get_r(self, y_pred):
        y = tf.numpy_function(self.get_r_, [y_pred], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_rnorm(self, y_pred, f_vec):
        y,w = tf.numpy_function(self.get_v_loss_rnorm_, [y_pred, f_vec], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_rnorm_batch(self, y_pred, f_vec):
        y,w = tf.numpy_function(self.get_v_loss_rnorm_batch_, [y_pred, f_vec], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_rnorm(self, y_pred, f_vec):
        y = tf.numpy_function(self.get_err_rnorm_, [y_pred, f_vec], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_rnorm_batch(self, y_pred, f_vec):
        y = tf.numpy_function(self.get_err_rnorm_batch_, [y_pred, f_vec], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_r_forces(self, y_pred, f_vec):
        y = tf.numpy_function(self.get_r_forces_, [y_pred, f_vec], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_wdiff(self, y_pred, b_true):
        y,w = tf.numpy_function(self.get_v_loss_wdiff_, [y_pred, b_true], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_welemdiff(self, y_pred, y_true, b_true):
        y,w = tf.numpy_function(self.get_v_loss_welemdiff_, [y_pred, y_true, b_true], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_wdiffdiff(self, y_pred, y_true, b_true):
        y,w = tf.numpy_function(self.get_v_loss_wdiffdiff_, [y_pred, y_true, b_true], (tf.float64, tf.float64))
        return y,w

    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_r_wdiff(self, y_pred, b_true):
        y = tf.numpy_function(self.get_r_wdiff_, [y_pred, b_true], (tf.float64))
        return y

    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_welemdiff(self, y_pred, y_true, b_true):
        y = tf.numpy_function(self.get_err_welemdiff_, [y_pred, y_true, b_true], (tf.float64))
        return y
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_wdiffdiff(self, y_pred, y_true, b_true):
        y = tf.numpy_function(self.get_err_wdiffdiff_, [y_pred, y_true, b_true], (tf.float64))
        return y
    
    def get_r_array(self, samples):
        b_list=[]
        for i, sample in enumerate(samples):
            b = self.get_r_(np.expand_dims(sample, axis=0))
            # We may need to remove the outer dimension
            b_list.append(np.array(b[0], copy=True))
        b_array=np.array(b_list)
        return b_array
    
    def get_r_forces_array(self, samples, forces):
        b_list=[]
        for i, sample in enumerate(samples):
            b = self.get_r_forces_(np.expand_dims(sample, axis=0), np.expand_dims(forces[i], axis=0))
            # We may need to remove the outer dimension
            b_list.append(np.array(b[0], copy=True))
        b_array=np.array(b_list)
        _ = self.get_r_forces_(np.expand_dims(samples[i], axis=0), 0*np.expand_dims(forces[i], axis=0)) #This is to reset the force conditions to 0 just in case
        return b_array
    
    def get_r_forces_withDirich_array(self, samples, forces):
        b_list=[]
        for i, sample in enumerate(samples):
            b = self.get_r_forces_withDirich_(np.expand_dims(sample, axis=0), np.expand_dims(forces[i], axis=0))
            # We may need to remove the outer dimension
            b_list.append(np.array(b[0], copy=True))
        b_array=np.array(b_list)
        _ = self.get_r_forces_withDirich_(np.expand_dims(samples[i], axis=0), 0*np.expand_dims(forces[i], axis=0)) #This is to reset the force conditions to 0 just in case
        return b_array
    