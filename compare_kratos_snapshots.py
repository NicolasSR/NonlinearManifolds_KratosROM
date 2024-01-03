import os
from collections import Counter

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator

from Utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, relative_l2_error_list, l2_error_list, mean_l2_error, forbenius_error

def calculate_X_norm_error(S_fom, S_rom):
    l2_error=mean_relative_l2_error(S_fom,S_rom)
    forb_error=relative_forbenius_error(S_fom,S_rom)
    print('X. Mean rel L2 error:', l2_error)
    print('X. Rel Forb. error:', forb_error)

def plot_rel_l2_errors(S_fom,S_rom):
    rel_l2_errors_list=relative_l2_error_list(S_fom,S_rom)
    plt.plot(rel_l2_errors_list)
    plt.semilogy()
    plt.xlabel('Step')
    plt.ylabel('Relative L2 error of displacements (nn vs fom)')
    plt.show()

def plot_l2_errors(x_true,x_pred):
    l2_errors_list=l2_error_list(x_true,x_pred)
    plt.plot(l2_errors_list)
    plt.semilogy()
    plt.xlabel('Step')
    plt.ylabel('L2 error of residuals')
    plt.show()

def calculate_R_norm_error(R_fom):
    l2_error=mean_l2_error(R_fom,0.0)
    forb_error=forbenius_error(R_fom,0.0)
    print('R. Mean L2 error:', l2_error)
    print('R. Forb. error:', forb_error)
    
def draw_x_error_abs_image(S_fom, S_rom):
    fig, (ax1) = plt.subplots(ncols=1)
    image=np.abs(S_rom-S_fom)
    im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],500,0], interpolation='none', cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=0, vmax=0.004, cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.015, vmax=0.03, cmap='jet')
    ax1.set_aspect(2)
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Displacement Abs Error')
    plt.show()

def draw_x_error_rel_image(S_fom, S_rom):
    fig, (ax1) = plt.subplots(ncols=1)
    image=np.abs((S_rom-S_fom))/(np.abs(S_fom)+1e-14)
    im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],500,0], interpolation='none', cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=0, vmax=1, cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.1, vmax=0.1, cmap='jet')
    ax1.set_aspect(2)
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Displacement Rel Error')
    plt.show()

def plot_error_over_parametric_space(S_fom, S_rom, F):
    rel_l2_errors_list=relative_l2_error_list(S_fom,S_rom)
    sc=plt.scatter(F[:,0],F[:,1], c=rel_l2_errors_list, s=4, cmap='jet', norm=matplotlib.colors.LogNorm())
    plt.colorbar(sc)
    plt.show()

if __name__ == "__main__":

    paths_list=[
        # 'saved_models_cantilever_big_range/POD/POD_Emb14',
        # 'Quad/Quad_least_squares_scale_global_Emb6',
        # 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[40, 40]_Emb6.20_LRsgdr0.001',
        # 'saved_models_cantilever_big_range/PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.001'
        'saved_models_cantilever_big_range/POD/POD_Emb6',
        'saved_models_cantilever_big_range/POD/POD_Emb8',
        'saved_models_cantilever_big_range/POD/POD_Emb10',
        'saved_models_cantilever_big_range/POD/POD_Emb12',
        'saved_models_cantilever_big_range/POD/POD_Emb14',
        'saved_models_cantilever_big_range/POD/POD_Emb16',
        'saved_models_cantilever_big_range/POD/POD_Emb18',
        'saved_models_cantilever_big_range/POD/POD_Emb20'
    ]

    # reference_snapshots_filename='datasets_two_forces_dense_extended/S_mu_dataset_300.npy'
    # reference_forces_filename='datasets_two_forces_dense_extended/F_mu_dataset_300.npy'
    # reference_snapshots_filename='datasets_rubber_hyperelastic_cantilever/FOM/FOM_equalForces_500steps.npy'
    # reference_forces_filename='datasets_rubber_hyperelastic_cantilever/FOM/POINTLOADS_equalForces_500steps.npy'

    reference_snapshots_filename='datasets_rubber_hyperelastic_cantilever_big_range/FOM/FOM_300steps_random_seed4.npy'
    reference_forces_filename='datasets_rubber_hyperelastic_cantilever_big_range/FOM/POINTLOADS_300steps_random_seed4.npy'

    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')

        # Get snapshots
        S_rom = np.load(model_path+'/NMROM_simulation_results_random300/ROM_snaps_converged_corrected.npy')
        # S_rom = np.load('datasets_two_forces_dense_extended/initial_states_dataset_err100000.npy')
        # S_rom = np.load(model_path+'/NMROM_simulation_results_3000steps_iter1/ROM_snaps_converged.npy')
        # S_rom = np.load(model_path+'/reconstruction_evaluation_results/recons_FOM_300steps_xneg_yne_bestr_.npy')
        S_fom = np.load(reference_snapshots_filename)

        print('Shape S_fom:', S_fom.shape)
        print('Shape S_rom:', S_rom.shape)

        print('Error norms')
        calculate_X_norm_error(S_fom, S_rom)

        plot_rel_l2_errors(S_fom, S_rom)

        draw_x_error_abs_image(S_fom, S_rom)
        draw_x_error_rel_image(S_fom, S_rom)


        print('======= Getting reactions matrix =======')

        with open(model_path+"/train_config.npy", "rb") as train_config_file:
            train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        dataset_path=train_config['dataset_path']

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation = StructuralMechanics_KratosSimulator('', train_config)

        F_FOM = np.load(reference_forces_filename)

        r_noForce = kratos_simulation.get_r_array(S_rom)
        np.save(model_path+'/NMROM_simulation_results_random300/ROM_residuals_noForce_converged_corrected.npy', r_noForce)

        r_force = kratos_simulation.get_r_forces_array(S_rom, F_FOM)
        np.save(model_path+'/NMROM_simulation_results_random300/ROM_residuals_converged_corrected.npy', r_force)
        # np.save(model_path+'/NMROM_simulation_results_3000steps_iter1/ROM_residuals_converged.npy', r_force)
        # np.save(model_path+'/reconstruction_evaluation_results/reactions_recons_FOM_300steps_xneg_yne_bestr_.npy', r_force)

        calculate_R_norm_error(r_force)
        # plot_l2_errors(r_force,0.0)