import json

import h5py
import numpy as np
import scipy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd

# import KratosMultiphysics as KMP
# import KratosMultiphysics.RomApplication as ROM
# import KratosMultiphysics.StructuralMechanicsApplication as SMA
# from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator
from Kratos_Simulators.fluid_dynamics_kratos_simulator import FluidDynamics_KratosSimulator

from sys import argv

def generate_finetune_datasets(dataset_path):

    S=np.load(dataset_path+'FOM.npy')
    R=np.load(dataset_path+'FOM_RESIDUALS.npy')
    F=np.load(dataset_path+'FOM_POINLOADS.npy')

    train_size=5000/S.shape[0]
    test_size=10000/S.shape[0]

    S_train, S_test, R_train, R_test, F_train, F_test = train_test_split(S,R,F, test_size=test_size, train_size=train_size, random_state=250)

    print('Shape S_train: ', S_train.shape)
    print('Shape S_test:', S_test.shape)
    print('Shape R_train:', R_train.shape)
    print('Shape R_test: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    with open(dataset_path+"S_finetune_train.npy", "wb") as f:
        np.save(f, S_train)
    with open(dataset_path+"S_finetune_test.npy", "wb") as f:
        np.save(f, S_test)
    with open(dataset_path+"R_finetune_train.npy", "wb") as f:
        np.save(f, R_train)
    with open(dataset_path+"R_finetune_test.npy", "wb") as f:
        np.save(f, R_test)
    with open(dataset_path+"F_finetune_train.npy", "wb") as f:
        np.save(f, F_train)
    with open(dataset_path+"F_finetune_test.npy", "wb") as f:
        np.save(f, F_test)

def generate_training_datasets(dataset_path):

    S=np.load(dataset_path+'FOM.npy')
    R=np.load(dataset_path+'FOM_RESIDUALS.npy')
    F=np.load(dataset_path+'FOM_POINLOADS.npy')

    test_size=1-20000/S.shape[0]

    S_train, S_test, R_train, R_test, F_train, F_test = train_test_split(S,R,F, test_size=test_size, random_state=274)

    print('Shape S_train: ', S_train.shape)
    print('Shape S_test:', S_test.shape)
    print('Shape R_train:', R_train.shape)
    print('Shape R_test: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    with open(dataset_path+"S_train.npy", "wb") as f:
        np.save(f, S_train)
    with open(dataset_path+"S_test.npy", "wb") as f:
        np.save(f, S_test)
    with open(dataset_path+"R_train.npy", "wb") as f:
        np.save(f, R_train)
    with open(dataset_path+"R_test.npy", "wb") as f:
        np.save(f, R_test)
    with open(dataset_path+"F_train.npy", "wb") as f:
        np.save(f, F_train)
    with open(dataset_path+"F_test.npy", "wb") as f:
        np.save(f, F_test)

def apply_random_noise(x_true, cropped_dof_ids):
        v=np.random.rand(x_true.shape[0]-len(cropped_dof_ids))
        v=v/np.linalg.norm(v)
        eps=np.random.rand()*1e-4
        v=v*eps
        x_app=x_true.copy()
        # print(x_app[4779:])
        x_app[~np.isin(np.arange(len(x_app)), cropped_dof_ids)]=x_app[~np.isin(np.arange(len(x_app)), cropped_dof_ids)]+v
        # print(x_app[4779:])

        return x_app

def generate_augm_finetune_datasets(dataset_path, kratos_simulation, augm_order):
    
    cropped_dof_ids = kratos_simulation.get_cropped_dof_ids()
    
    with open(dataset_path+"S_finetune_train.npy", "rb") as f:
        S_train=np.load(f)
    with open(dataset_path+"F_finetune_train.npy", "rb") as f:
        F_train=np.load(f)
    with open(dataset_path+"R_finetune_noF_train.npy", "rb") as f:
        R_train=np.load(f)

    S_augm=[]
    F_augm=[]
    R_augm=[]

    for i in range(S_train.shape[0]):
        S_augm.append(S_train[i])
        F_augm.append(F_train[i])
        R_augm.append(R_train[i])
        for n in range(augm_order):
            s_noisy = apply_random_noise(S_train[i], cropped_dof_ids)
            r_noisy = kratos_simulation.get_r_(np.expand_dims(s_noisy, axis=0))[0]
            S_augm.append(s_noisy.copy())
            F_augm.append(F_train[i])
            R_augm.append(r_noisy)

        if i%100 == 0:
            print('Iteration: ', i, 'of ', S_train.shape[0], '. Current length: ', len(S_augm))
    
    S_augm=np.array(S_augm)
    F_augm=np.array(F_augm)
    R_augm=np.array(R_augm)

    with open(dataset_path+"S_augm_train.npy", "wb") as f:
        np.save(f, S_augm)
    # with open(dataset_path+"R_augm_train.npy", "wb") as f:
    with open(dataset_path+"R_augm_noF_train.npy", "wb") as f:
        np.save(f, R_augm)
    with open(dataset_path+"F_augm_train.npy", "wb") as f:
        np.save(f, F_augm)

def generate_augm_finetune_dataset_soriginal(dataset_path, kratos_simulation, augm_order):
    
    with open(dataset_path+"S_finetune_train.npy", "rb") as f:
        S_train=np.load(f)

    S_augm_orig=[]

    for i in range(S_train.shape[0]):
        S_augm_orig.append(S_train[i])
        for n in range(augm_order):
            S_augm_orig.append(S_train[i])

        if i%100 == 0:
            print('Iteration: ', i, 'of ', S_train.shape[0], '. Current length: ', len(S_augm_orig))
    
    S_augm_orig=np.array(S_augm_orig)

    with open(dataset_path+"S_augm_original_train.npy", "wb") as f:
        np.save(f, S_augm_orig)

def generate_residuals_noforce(dataset_path, kratos_simulation):
    
    with open(dataset_path+"S_test_small.npy", "rb") as f:
        S_fom=np.load(f)
        print(S_fom.shape)

    # R_noF=kratos_simulation.get_r_batch_noDirich_(S_fom)
    R_noF=kratos_simulation.get_r_batch_(S_fom)
    

    # with open(dataset_path+"R_train_noDirich.npy", "wb") as f:
    with open(dataset_path+"R_test_small.npy", "wb") as f:
        np.save(f, R_noF)

def join_datasets(dataset_path):

    # S1=np.load(dataset_path+"fom_snapshots_1.npy").T
    # S2=np.load(dataset_path+"fom_snapshots_2.npy").T
    # S3=np.load(dataset_path+"fom_snapshots_3.npy").T
    # S4=np.load(dataset_path+"fom_snapshots_4.npy").T
    # S5=np.load(dataset_path+"fom_snapshots_5.npy").T
    # S6=np.load(dataset_path+"fom_snapshots_6.npy").T
    # S7=np.load(dataset_path+"fom_snapshots_7.npy").T
    # S8=np.load(dataset_path+"fom_snapshots_8.npy").T
    # S9=np.load(dataset_path+"fom_snapshots_9.npy").T
    # S10=np.load(dataset_path+"fom_snapshots_10.npy").T

    # Stest1=np.load(dataset_path+"fom_snapshots_test_1.npy").T
    # Stest2=np.load(dataset_path+"fom_snapshots_test_2.npy").T

    Stest1=np.load(dataset_path+"fom_snapshots_linear.npy").T


    # S=np.concatenate([S1,S2,S3,S4,S5,S6,S7,S8,S9,S10], axis=0)
    S=np.concatenate([Stest1], axis=0)
    # S=np.concatenate([S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,Stest1,Stest2], axis=0)
    np.save(dataset_path+"S_test_linear.npy", S)
    print(S.shape)

if __name__ == "__main__":

    # train_config = {
    #     "nn_type": 'standard_config', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
    #     "name": 'standard_config',
    #     "dataset_path": 'datasets_fluid_past_cylinder/',
    #     "project_parameters_file":'ProjectParameters_FOM.json',
    #     "use_force":False
    #  }
    train_config = {
        "nn_type": 'standard_config', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'standard_config',
        "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
        "project_parameters_file":'ProjectParameters_FOM.json',
        "use_force":False
     }

    dataset_path=train_config["dataset_path"]

    # Create a fake Analysis stage to calculate the predicted residuals
    working_path=argv[1]+"/"
    needs_truncation=False
    residual_scale_factor=1.0
    kratos_simulation = StructuralMechanics_KratosSimulator(working_path, train_config)
    # kratos_simulation = FluidDynamics_KratosSimulator(working_path, train_config)

    # generate_training_datasets(dataset_path)
    # generate_finetune_datasets(dataset_path)
    # generate_augm_finetune_datasets(dataset_path, kratos_simulation, 3)
    # generate_augm_finetune_dataset_soriginal(dataset_path, kratos_simulation, 3)
    generate_residuals_noforce(dataset_path, kratos_simulation)
    # generate_residuals_noforce('', kratos_simulation)
    # join_datasets(dataset_path)