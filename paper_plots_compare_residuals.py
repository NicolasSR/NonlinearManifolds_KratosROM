import argparse

import numpy as np
from matplotlib import pyplot as plt

from Utils.custom_metrics import mean_l2_error, mean_relative_l2_error, forbenius_error, relative_forbenius_error
from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator


def plot_rRorce_error(model_info_list):

    PODANN_sLoss_recons = []
    PODANN_sLoss_nmrom = []
    PODANN_sLoss_q = []

    PODANN_rLoss_recons = []
    PODANN_rLoss_nmrom = []
    PODANN_rLoss_q = []

    POD_recons = []
    POD_nmrom = []
    POD_q = []
    
    for model_info in model_info_list:
        if model_info.label=="rLoss":
            PODANN_rLoss_recons.append(model_info.recons_rForce_frobErr)
            PODANN_rLoss_nmrom.append(model_info.nmrom_rForce_frobErr)
            PODANN_rLoss_q.append(model_info.q_inf)
        elif model_info.label=="sLoss":
            PODANN_sLoss_recons.append(model_info.recons_rForce_frobErr)
            PODANN_sLoss_nmrom.append(model_info.nmrom_rForce_frobErr)
            PODANN_sLoss_q.append(model_info.q_inf)
        elif model_info.label=="POD":
            POD_recons.append(model_info.recons_rForce_frobErr)
            POD_nmrom.append(model_info.nmrom_rForce_frobErr)
            POD_q.append(model_info.q_inf)

    plt.plot(PODANN_sLoss_q, PODANN_sLoss_recons, '--b', marker='o', markersize=4, label='sLoss Proj.')
    plt.plot(PODANN_sLoss_q, PODANN_sLoss_nmrom, '-b', marker='+', markersize=4, label='sLoss NMROM')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_recons, '--r', marker='*', markersize=4, label='rLoss Proj.')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_nmrom, '-r', marker='2', markersize=4, label='rLoss NMROM')
    plt.plot(POD_q, POD_recons, '--g', marker='D', markersize=4, label='POD Proj.')
    plt.plot(POD_q, POD_nmrom, '-g', marker='X', markersize=4, label='POD NMROM')

    plt.semilogy()
    plt.legend()
    plt.grid(which='major')
    plt.xlabel(r'$q_{inf}$')
    plt.ylabel(r'$e_{r,\mathrm{frob}}$')

    plt.show()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    args = parser.parse_args()
    working_path = args.working_path+'/'

    train_config = {
        "nn_type": 'standard_config', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'standard_config',
        "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
        "project_parameters_file":'ProjectParameters_FOM.json',
        "use_force":False
     }
    kratos_simulation = StructuralMechanics_KratosSimulator(working_path, train_config)

    id=141

    S = np.load('datasets_rubber_hyperelastic_cantilever_big_range/S_test_small.npy')[id]
    R_noForce = np.load('datasets_rubber_hyperelastic_cantilever_big_range/R_test_small.npy')[id]
    F = np.load('datasets_rubber_hyperelastic_cantilever_big_range/F_test_small.npy')[id]
    F_parsed = [F[0,0],F[6,1]]
    print('Forces applied: ', F_parsed)


    R_Force=kratos_simulation.get_r_forces_batch_(np.expand_dims(S,axis=0),np.expand_dims(F,axis=0))[0]
    # print(R_Force)

    plt.plot(np.abs(R_noForce))
    plt.plot(np.abs(R_Force))
    plt.semilogy()
    plt.show()


    print('FINISHED EVALUATING')







# rLoss_galerkin_uErr = [0.00016570047444816174, 7.528659101547164e-08]
# sLoss_galerkin_uErr = [1.0, 1.921410464249454e-05]

# rLoss_galerkin_rErr = [54.270210957191814, 0.10554194886166827]
# sLoss_galerkin_rErr = [1000.0, 21.95635276814271]

# sLoss_reconstr_uErr = [5.0965800572956246e-05, 1.1569986325000624e-06]
# rLoss_reconstr_uErr = [4.367536710106586e-05, 7.975046860941974e-09]

# sLoss_reconstr_rErr = [2472.3903407886028,160.89785468030206]
# rLoss_reconstr_rErr = [296.8093214558384, 0.620354890020508]

# POD_recons_uErr = [0.0009358944254847696, 5.7e-4, 3.6e-4, 2.3e-4, 1.7e-4, 1.0e-4, 3.4e-5, 2.3e-5, 1.0362969606677837e-06]
# POD_recons_rErr = [7992.032653971569, 35.04313196375438]

# k_vec=[6,20]
# POD_k_vec = [6,7,8,9,10,11,12,13,20]


# plt.plot(k_vec, sLoss_galerkin_uErr, 'o', markersize=4, label="ANN-PROM SLoss")
# plt.plot(k_vec, sLoss_reconstr_uErr, 'o', markersize=4, label="Reconstruction SLoss")
# plt.plot(k_vec, rLoss_reconstr_uErr, 'o', markersize=4, label="Reconstruction RLoss")
# plt.plot(k_vec, rLoss_galerkin_uErr, 'o', markersize=4, label="ANN-PROM RLoss")
# plt.plot(POD_k_vec, POD_recons_uErr, 'o', markersize=4, label="Reconstruction POD")
# plt.semilogy()
# plt.legend()
# plt.show()


# plt.plot(k_vec, sLoss_galerkin_rErr, 'o', markersize=4, label="ANN-PROM SLoss")
# plt.plot(k_vec, sLoss_reconstr_rErr, 'o', markersize=4, label="Reconstruction SLoss")
# plt.plot(k_vec, rLoss_reconstr_rErr, 'o', markersize=4, label="Reconstruction RLoss")
# plt.plot(k_vec, rLoss_galerkin_rErr, 'o', markersize=4, label="ANN-PROM RLoss")
# plt.plot(k_vec, POD_recons_rErr, 'o', markersize=4, label="Reconstruction POD")
# plt.semilogy()
# plt.legend()
# plt.show()



