import numpy as np
import scipy

import tensorflow as tf

import KratosMultiphysics as KMP
import KratosMultiphysics.FluidDynamicsApplication as FDA

from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

import time

class FluidDynamics_KratosSimulator():

    def __init__(self, working_path, train_config):
        if "project_parameters_file" in train_config:
            project_parameters_path=train_config["dataset_path"]+train_config["project_parameters_file"]
        else: 
            project_parameters_path='ProjectParameters_fom.json'
            print('LOADED DEFAULT PROJECT PARAMETERS FILE')
        with open(working_path+project_parameters_path, 'r') as parameter_file:
            parameters = KMP.Parameters(parameter_file.read())

        global_model = KMP.Model()
        self.fake_simulation = FluidDynamicsAnalysis(global_model, parameters)
        self.fake_simulation.Initialize()
        self.fake_simulation.InitializeSolutionStep()

        self.space = KMP.UblasSparseSpace()
        self.strategy = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        self.buildsol = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        self.scheme = self.fake_simulation._GetSolver()._GetScheme()
        self.modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        self.var_utils = KMP.VariableUtils()


    # def get_crop_matrix(self):
    #     indices=[]

    #     inlet_modelpart = self.modelpart.GetModel().GetModelPart('FluidModelPart').GetSubModelPart('AutomaticInlet2D_Inlet')
    #     outlet_modelpart = self.modelpart.GetModel().GetModelPart('FluidModelPart').GetSubModelPart('Outlet2D_Outlet')

    #     nodes_inlet = inlet_modelpart.Nodes
    #     nodes_outlet = outlet_modelpart.Nodes

    #     for node in nodes_inlet:
    #         indices.append([(node.Id-1)*3,(node.Id-1)*3])
    #         indices.append([(node.Id-1)*3+1,(node.Id-1)*3+1])

    #     for node in nodes_outlet:
    #         indices.append([(node.Id-1)*3+2,(node.Id-1)*3+2])

    #     num_rows=self.modelpart.NumberOfNodes()*3
        
    #     values=np.ones(num_rows)
    #     crop_mat_tf = tf.sparse.SparseTensor(
    #         indices=indices,
    #         values=values,
    #         dense_shape=[num_rows,num_rows])
    #     indices=np.asarray(indices)
    #     crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_rows]).tocsr()
        
    #     return crop_mat_tf, crop_mat_scp
    
    def get_crop_matrix(self):
        indices=[]

        for node in self.modelpart.Nodes:
            dof = node.GetDof(KMP.PRESSURE)
            if dof.IsFixed():
                indices.append([dof.EquationId,dof.EquationId])

            dof = node.GetDof(KMP.VELOCITY_X)
            if dof.IsFixed():
                indices.append([dof.EquationId,dof.EquationId])

            dof = node.GetDof(KMP.VELOCITY_Y)
            if dof.IsFixed():
                indices.append([dof.EquationId,dof.EquationId])

        num_rows=self.modelpart.NumberOfNodes()*3
        
        values=np.ones(len(indices))
        crop_mat_tf = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_rows,num_rows])
        indices=np.asarray(indices)
        crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_rows]).tocsr()
        
        return crop_mat_tf, crop_mat_scp
            

    def get_cropped_dof_ids(self):
        fixed_dofs=[]

        for node in self.modelpart.Nodes:
            dof = node.GetDof(KMP.PRESSURE)
            if dof.IsFixed():
                fixed_dofs.append([dof.EquationId,dof.EquationId])

            dof = node.GetDof(KMP.VELOCITY_X)
            if dof.IsFixed():
                fixed_dofs.append([dof.EquationId,dof.EquationId])

            dof = node.GetDof(KMP.VELOCITY_Y)
            if dof.IsFixed():
                fixed_dofs.append([dof.EquationId,dof.EquationId])

        return fixed_dofs

    def project_prediction_vectorial_optim_batch(self, y_pred):
        num_nodes=self.modelpart.NumberOfNodes()

        values = y_pred.reshape((num_nodes,3))
        values_p = values[:,0].reshape((num_nodes)).copy()
        values_v = values[:,1:].reshape((num_nodes*2)).copy()

        nodes_array=self.modelpart.Nodes
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.PRESSURE, values_p, 0)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.VELOCITY, values_v, 0)

        self.fake_simulation._GetSolver().GetComputingModelPart().ProcessInfo.SetValue(KMP.BDF_COEFFICIENTS,[0,0,0])

    
    def get_v_loss_rdiff_batch_(self, y_pred, b_true):
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        xD  = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(xD, self.space.Size(b))
        foo  = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(foo, self.space.Size(b))

        err_r_list=[]
        v_loss_r_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)
            self.space.SetToZeroVector(xD)
            self.space.SetToZeroVector(foo)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)
            # self.buildsol.ApplyDirichletConditions(self.scheme,self.modelpart,A,xD,foo)

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
    
    def get_err_rdiff_batch_(self, y_pred, b_true):
        b_pred = self.get_r_batch_(y_pred)
        # b_pred = self.get_r_batch_noDirich_(y_pred)
        err_r = b_pred - b_true
        return err_r
    
    def get_r_batch_(self, y_pred):

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        # xD  = self.space.CreateEmptyVectorPointer()
        # self.space.ResizeVector(xD, self.space.Size(b))

        b_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)
            # self.space.SetToZeroVector(xD)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)
            # self.buildsol.ApplyDirichletConditions(self.scheme,self.modelpart,A,xD,b)

            b_list.append(np.expand_dims(np.array(b, copy=True),axis=0))

        b_batch = np.concatenate(b_list, axis = 0)
        
        return b_batch
    
    def get_A_e_vec_batch_(self, y_true, err):
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        # xD  = self.space.CreateEmptyVectorPointer()
        # self.space.ResizeVector(xD, self.space.Size(b))
        # foo  = self.space.CreateEmptyVectorPointer()
        # self.space.ResizeVector(foo, self.space.Size(b))

        err_r_list=[]
        v_loss_r_list=[]

        for i in range(y_true.shape[0]):
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)
            # self.space.SetToZeroVector(xD)
            # self.space.SetToZeroVector(foo)

            self.project_prediction_vectorial_optim_batch(y_true[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)
            # self.buildsol.ApplyDirichletConditions(self.scheme,self.modelpart,A,xD,foo)

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
    
    def get_A_noDirich_(self, y_pred):

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        xD  = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(xD, self.space.Size(b))

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)
        self.space.SetToZeroVector(xD)

        self.project_prediction_vectorial_optim_batch(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        self.buildsol.ApplyDirichletConditions(self.scheme,self.modelpart,A,xD,b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))
        
        return Ascipy

    def get_r_batch_noDirich_(self, y_pred):

        b = self.strategy.GetSystemVector()

        b_list=[]

        for i in range(y_pred.shape[0]):
            self.space.SetToZeroVector(b)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.BuildRHS(self.scheme, self.modelpart, b)

            b_list.append(np.expand_dims(np.array(b, copy=True),axis=0))

        b_batch = np.concatenate(b_list, axis = 0)
        
        return b_batch
    
    def get_r_(self, y_pred):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim_batch(y_pred[0])

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        
        return b
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_rdiff_batch(self, y_pred, b_true):
        y,w = tf.numpy_function(self.get_v_loss_rdiff_batch_, [y_pred, b_true], (tf.float64, tf.float64))
        return y,w
    
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_err_rdiff_batch(self, y_pred, b_true):
        y = tf.numpy_function(self.get_err_rdiff_batch_, [y_pred, b_true], (tf.float64))
        return y
    
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
    def get_r(self, y_pred):
        y = tf.numpy_function(self.get_r_, [y_pred], (tf.float64))
        return y
    
    def get_r_array(self, samples):
        b_list=[]
        for i, sample in enumerate(samples):
            b = self.get_r_(np.expand_dims(sample, axis=0))
            # We may need to remove the outer dimension
            b_list.append(np.array(b[0], copy=True))
        b_array=np.array(b_list)
        return b_array