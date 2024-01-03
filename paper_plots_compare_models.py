import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

from Utils.custom_metrics import mean_l2_error, mean_relative_l2_error, forbenius_error, relative_forbenius_error



class Model_Info:
    def __init__(self, working_path, model_path, label, q_inf, recons_s_ref, nmrom_s_ref, recons_r_ref, nmrom_r_ref):

        self.label=label
        self.q_inf=q_inf

        self.working_path = working_path
        self.model_path = working_path+'saved_models_cantilever_big_range/'+model_path

        self.recons_s_path = None
        self.recons_rForce_path = None
        self.recons_rNoForce_path = None
        self.nmrom_s_path = None
        self.nmrom_rForce_path = None
        self.nmrom_rNoForce_path = None

        self.recons_s_frobErr = None
        self.recons_rForce_frobErr = None
        self.recons_rNoForce_frobErr = None
        self.nmrom_s_frobErr = None
        self.nmrom_rForce_frobErr = None
        self.nmrom_rNoForce_frobErr = None

        self.non_converged_samples = 0

        if "tf_ronly" in model_path:
            best_type_decorator = '_bestr_'
        elif "tf_sonly" in model_path:
            best_type_decorator = '_bestx_'
        else:
            best_type_decorator = ''
        
        test_type_decorator = '_test_small_'

        self.get_results_paths(best_type_decorator, test_type_decorator)

        self.get_frob_results(recons_s_ref, nmrom_s_ref, recons_r_ref, nmrom_r_ref)

    def get_results_paths(self, best_type_decorator, test_type_decorator):
        self.recons_s_path = self.model_path+'reconstruction_evaluation_results/s_pred_matrix'+best_type_decorator+test_type_decorator+'.npy'
        self.recons_rForce_path = self.model_path+'reconstruction_evaluation_results/r_force_pred_matrix'+best_type_decorator+test_type_decorator+'.npy'
        self.recons_rNoForce_path = self.model_path+'reconstruction_evaluation_results/r_noForce_pred_matrix'+best_type_decorator+test_type_decorator+'.npy'
        self.nmrom_s_path = self.model_path+'NMROM_simulation_results_random300/ROM_snaps_converged_corrected.npy'
        self.nmrom_rForce_path = self.model_path+'NMROM_simulation_results_random300/ROM_residuals_converged_corrected.npy'
        self.nmrom_rNoForce_path = self.model_path+'NMROM_simulation_results_random300/ROM_residuals_noForce_converged_corrected.npy'

    def get_frob_results(self, recons_s_ref, nmrom_s_ref, recons_r_ref, nmrom_r_ref):
        nmrom_s=np.load(self.nmrom_s_path)
        nan_rows = np.unique(np.argwhere(np.isnan(nmrom_s))[:,0])
        self.non_converged_samples=nan_rows.shape[0]
        self.recons_s_frobErr = relative_forbenius_error(recons_s_ref,np.load(self.recons_s_path))
        self.recons_rForce_frobErr = forbenius_error(np.load(self.recons_rForce_path),0.0)
        self.recons_rNoForce_frobErr = relative_forbenius_error(recons_r_ref,np.load(self.recons_rNoForce_path))
        self.nmrom_s_frobErr = relative_forbenius_error(np.delete(nmrom_s_ref, nan_rows, axis=0),np.delete(nmrom_s, nan_rows, axis=0))
        self.nmrom_rForce_frobErr = forbenius_error(np.delete(np.load(self.nmrom_rForce_path), nan_rows, axis=0),0.0)
        self.nmrom_rNoForce_frobErr = relative_forbenius_error(np.delete(nmrom_r_ref, nan_rows, axis=0), np.delete(np.load(self.nmrom_rNoForce_path), nan_rows, axis=0))

def plot_s_error(model_info_list):

    PODANN_sLoss_recons = []
    PODANN_sLoss_nmrom = []
    PODANN_sLoss_q = []

    PODANN_rLoss_recons = []
    PODANN_rLoss_nmrom = []
    PODANN_rLoss_q = []

    PODANN_rLoss_fine_recons = []
    PODANN_rLoss_fine_nmrom = []
    PODANN_rLoss_fine_q = []

    POD_recons = []
    POD_nmrom = []
    POD_q = []
    
    for model_info in model_info_list:
        if model_info.label=="rLoss":
            PODANN_rLoss_recons.append(model_info.recons_s_frobErr)
            PODANN_rLoss_nmrom.append(model_info.nmrom_s_frobErr)
            PODANN_rLoss_q.append(model_info.q_inf)
        elif model_info.label=="rLoss_finetuned":
            PODANN_rLoss_fine_recons.append(model_info.recons_s_frobErr)
            PODANN_rLoss_fine_nmrom.append(model_info.nmrom_s_frobErr)
            PODANN_rLoss_fine_q.append(model_info.q_inf)
        elif model_info.label=="sLoss":
            PODANN_sLoss_recons.append(model_info.recons_s_frobErr)
            PODANN_sLoss_nmrom.append(model_info.nmrom_s_frobErr)
            PODANN_sLoss_q.append(model_info.q_inf)
        elif model_info.label=="POD":
            POD_recons.append(model_info.recons_s_frobErr)
            POD_nmrom.append(model_info.nmrom_s_frobErr)
            POD_q.append(model_info.q_inf)

    fig = plt.figure()

    plt.plot(PODANN_sLoss_q, PODANN_sLoss_recons, '--b', marker='o', markersize=4, label='sLoss Proj.')
    plt.plot(PODANN_sLoss_q, PODANN_sLoss_nmrom, '-b', marker='+', markersize=4, label='sLoss NMROM')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_recons, '--r', marker='*', markersize=4, label='rLoss Proj.')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_nmrom, '-r', marker='2', markersize=4, label='rLoss NMROM')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_recons, '--k', marker='*', markersize=4, label='rLoss_fine Proj.')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_nmrom, '-k', marker='2', markersize=4, label='rLoss_fine NMROM')
    plt.plot(POD_q, POD_recons, '--g', marker='D', markersize=4, label='POD Proj.')
    plt.plot(POD_q, POD_nmrom, '-g', marker='X', markersize=4, label='POD NMROM')

    plt.semilogy()

    ax = plt.gca()

    for model_info in model_info_list:
        if model_info.non_converged_samples > 0:
            xy=(model_info.q_inf, model_info.nmrom_s_frobErr)
            radius=0.02

            # Calculate figure dimension ratio width/height
            pr = fig.get_figwidth()/fig.get_figheight()

            # Get the transScale (important if one of the axis is in log-scale)
            tscale = ax.transScale + (ax.transLimits + ax.transAxes)
            ctscale = tscale.transform_point(xy)
            cfig = fig.transFigure.inverted().transform(ctscale)

            circ = patches.Ellipse(cfig, radius, radius*pr,transform=fig.transFigure, fill=False)

            # Draw circle
            ax.add_patch(circ)

    ax2=ax.twinx()

    ax2.plot([], [], label= 'Non-converged samples', marker='o', markersize=8, 
         markeredgecolor='k', markerfacecolor='w', linestyle='')
    ax2.get_yaxis().set_visible(False)

    ax.legend()
    ax2.legend(loc=3)
    ax.grid(which='major')
    ax.set_xlabel(r'$q_{inf}$')
    ax.set_ylabel(r'$e_{u}$')
    ax2.grid(which='major')

    plt.show()


def plot_rNoRorce_error(model_info_list):

    PODANN_sLoss_recons = []
    PODANN_sLoss_nmrom = []
    PODANN_sLoss_q = []

    PODANN_rLoss_recons = []
    PODANN_rLoss_nmrom = []
    PODANN_rLoss_q = []
    
    PODANN_rLoss_fine_recons = []
    PODANN_rLoss_fine_nmrom = []
    PODANN_rLoss_fine_q = []

    POD_recons = []
    POD_nmrom = []
    POD_q = []
    
    for model_info in model_info_list:
        if model_info.label=="rLoss":
            PODANN_rLoss_recons.append(model_info.recons_rNoForce_frobErr)
            PODANN_rLoss_nmrom.append(model_info.nmrom_rNoForce_frobErr)
            PODANN_rLoss_q.append(model_info.q_inf)
        elif model_info.label=="rLoss_finetuned":
            PODANN_rLoss_fine_recons.append(model_info.recons_rNoForce_frobErr)
            PODANN_rLoss_fine_nmrom.append(model_info.nmrom_rNoForce_frobErr)
            PODANN_rLoss_fine_q.append(model_info.q_inf)
        elif model_info.label=="sLoss":
            PODANN_sLoss_recons.append(model_info.recons_rNoForce_frobErr)
            PODANN_sLoss_nmrom.append(model_info.nmrom_rNoForce_frobErr)
            PODANN_sLoss_q.append(model_info.q_inf)
        elif model_info.label=="POD":
            POD_recons.append(model_info.recons_rNoForce_frobErr)
            POD_nmrom.append(model_info.nmrom_rNoForce_frobErr)
            POD_q.append(model_info.q_inf)

    fig = plt.figure()

    plt.plot(PODANN_sLoss_q, PODANN_sLoss_recons, '--b', marker='o', markersize=4, label='sLoss Proj.')
    plt.plot(PODANN_sLoss_q, PODANN_sLoss_nmrom, '-b', marker='+', markersize=4, label='sLoss NMROM')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_recons, '--r', marker='*', markersize=4, label='rLoss Proj.')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_nmrom, '-r', marker='2', markersize=4, label='rLoss NMROM')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_recons, '--k', marker='*', markersize=4, label='rLoss_fine Proj.')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_nmrom, '-k', marker='2', markersize=4, label='rLoss_fine NMROM')
    plt.plot(POD_q, POD_recons, '--g', marker='D', markersize=4, label='POD Proj.')
    plt.plot(POD_q, POD_nmrom, '-g', marker='X', markersize=4, label='POD NMROM')

    plt.semilogy()

    ax = plt.gca()

    for model_info in model_info_list:
        if model_info.non_converged_samples > 0:
            xy=(model_info.q_inf, model_info.nmrom_rForce_frobErr)
            radius=0.02

            # Calculate figure dimension ratio width/height
            pr = fig.get_figwidth()/fig.get_figheight()

            # Get the transScale (important if one of the axis is in log-scale)
            tscale = ax.transScale + (ax.transLimits + ax.transAxes)
            ctscale = tscale.transform_point(xy)
            cfig = fig.transFigure.inverted().transform(ctscale)

            circ = patches.Ellipse(cfig, radius, radius*pr,transform=fig.transFigure, fill=False)

            # Draw circle
            ax.add_patch(circ)

    ax2=ax.twinx()

    ax2.plot([], [], label= 'Non-converged samples', marker='o', markersize=8, 
         markeredgecolor='k', markerfacecolor='w', linestyle='')
    ax2.get_yaxis().set_visible(False)

    ax.legend()
    ax2.legend(loc=3)
    ax.grid(which='major')
    ax.set_xlabel(r'$q_{inf}$')
    ax.set_ylabel(r'$e_{r,\mathrm{diff}}$')
    ax2.grid(which='major')
    plt.show()

def plot_rRorce_error(model_info_list):

    PODANN_sLoss_recons = []
    PODANN_sLoss_nmrom = []
    PODANN_sLoss_q = []

    PODANN_rLoss_recons = []
    PODANN_rLoss_nmrom = []
    PODANN_rLoss_q = []

    PODANN_rLoss_fine_recons = []
    PODANN_rLoss_fine_nmrom = []
    PODANN_rLoss_fine_q = []

    POD_recons = []
    POD_nmrom = []
    POD_q = []
    
    for model_info in model_info_list:
        if model_info.label=="rLoss":
            PODANN_rLoss_recons.append(model_info.recons_rForce_frobErr)
            PODANN_rLoss_nmrom.append(model_info.nmrom_rForce_frobErr)
            PODANN_rLoss_q.append(model_info.q_inf)
        if model_info.label=="rLoss_fine":
            PODANN_rLoss_fine_recons.append(model_info.recons_rForce_frobErr)
            PODANN_rLoss_fine_nmrom.append(model_info.nmrom_rForce_frobErr)
            PODANN_rLoss_fine_q.append(model_info.q_inf)
        elif model_info.label=="sLoss":
            PODANN_sLoss_recons.append(model_info.recons_rForce_frobErr)
            PODANN_sLoss_nmrom.append(model_info.nmrom_rForce_frobErr)
            PODANN_sLoss_q.append(model_info.q_inf)
        elif model_info.label=="POD":
            POD_recons.append(model_info.recons_rForce_frobErr)
            POD_nmrom.append(model_info.nmrom_rForce_frobErr)
            POD_q.append(model_info.q_inf)

    fig = plt.figure()

    plt.plot(PODANN_sLoss_q, PODANN_sLoss_recons, '--b', marker='o', markersize=4, label='sLoss Proj.')
    plt.plot(PODANN_sLoss_q, PODANN_sLoss_nmrom, '-b', marker='+', markersize=4, label='sLoss NMROM')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_recons, '--r', marker='*', markersize=4, label='rLoss Proj.')
    plt.plot(PODANN_rLoss_q, PODANN_rLoss_nmrom, '-r', marker='2', markersize=4, label='rLoss NMROM')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_recons, '--k', marker='*', markersize=4, label='rLoss_fine Proj.')
    plt.plot(PODANN_rLoss_fine_q, PODANN_rLoss_fine_nmrom, '-k', marker='2', markersize=4, label='rLoss_fine NMROM')
    plt.plot(POD_q, POD_recons, '--g', marker='D', markersize=4, label='POD Proj.')
    plt.plot(POD_q, POD_nmrom, '-g', marker='X', markersize=4, label='POD NMROM')

    plt.semilogy()

    ax = plt.gca()

    for model_info in model_info_list:
        if model_info.non_converged_samples > 0:
            xy=(model_info.q_inf, model_info.nmrom_rForce_frobErr)
            radius=0.02

            # Calculate figure dimension ratio width/height
            pr = fig.get_figwidth()/fig.get_figheight()

            # Get the transScale (important if one of the axis is in log-scale)
            tscale = ax.transScale + (ax.transLimits + ax.transAxes)
            ctscale = tscale.transform_point(xy)
            cfig = fig.transFigure.inverted().transform(ctscale)

            circ = patches.Ellipse(cfig, radius, radius*pr,transform=fig.transFigure, fill=False)

            # Draw circle
            ax.add_patch(circ)

    ax2=ax.twinx()

    ax2.plot([], [], label= 'Non-converged samples', marker='o', markersize=8, 
         markeredgecolor='k', markerfacecolor='w', linestyle='')
    ax2.get_yaxis().set_visible(False)

    ax.legend()
    ax2.legend(loc=3)
    ax.grid(which='major')
    ax.set_xlabel(r'$q_{inf}$')
    ax.set_ylabel(r'$e_{r,\mathrm{norm}}$')
    ax2.grid(which='major')
    plt.show()


def print_parametric_space():
    samples=np.array([[ 1265.98946765, -2656.01535315],
 [-1734.01053235,  -656.01535315],
 [ 2765.98946765,  1343.98464685],
 [ -234.01053235, -1322.68201982],
 [  515.98946765,   677.31798018],
 [-2484.01053235,  2677.31798018]])
    plt.scatter(samples[:,0],samples[:,1])
    plt.xlim([-3000,3000])
    plt.ylim([-3000,3000])
    plt.xlabel("Line load x [N/m]")
    plt.ylabel("Line load y [N/m]")
    plt.gca().set_aspect('equal')
    plt.show()


if __name__=="__main__":

    # plt.rcParams['text.usetex'] = True

    # FOM=np.load('datasets_rubber_hyperelastic_cantilever_big_range/FOM.npy')
    # print(FOM.shape)
    # _,sigma,_=np.linalg.svd(FOM.T)

    # sig_acum=[]
    # for i in range(sigma.shape[0]):
    #     sig_acum.append(1-sum(sigma[:i+1])/sum(sigma))

    # plt.plot(sig_acum[:200])
    # # plt.plot(sigma)
    # plt.semilogy()
    # plt.xlabel('Singular value index j')
    # plt.ylabel(r'$1-\sum^j_{i=1}\sigma_i/\sum^{1594}_{i=1}\sigma_i$')
    # plt.grid(which='major')
    # plt.show()

    # print_parametric_space()

    # exit()

    result_cases = [{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 6
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb8.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 8
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb10.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 10
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb12.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 12
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb14.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 14
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb16.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 16
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb18.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 18
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001/',
        "label": 'rLoss',
        "q_inf": 20
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 6
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb8.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 8
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb10.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 10
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb12.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 12
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb14.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 14
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb16.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 16
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb18.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 18
     },{
        "model_path": 'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.0001/',
        "label": 'rLoss_finetuned',
        "q_inf": 20
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 6
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb8.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 8
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb10.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 10
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb12.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 12
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb14.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 14
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb16.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 16
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb18.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 18
     },{
        "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001/',
        "label": 'sLoss',
        "q_inf": 20
     },{
        "model_path": 'POD/POD_Emb6/',
        "label": 'POD',
        "q_inf": 6
     },{
        "model_path": 'POD/POD_Emb8/',
        "label": 'POD',
        "q_inf": 8
     },{
        "model_path": 'POD/POD_Emb10/',
        "label": 'POD',
        "q_inf": 10
     },{
        "model_path": 'POD/POD_Emb12/',
        "label": 'POD',
        "q_inf": 12
     },{
        "model_path": 'POD/POD_Emb14/',
        "label": 'POD',
        "q_inf": 14
     },{
        "model_path": 'POD/POD_Emb16/',
        "label": 'POD',
        "q_inf": 16
     },{
        "model_path": 'POD/POD_Emb18/',
        "label": 'POD',
        "q_inf": 18
     },{
        "model_path": 'POD/POD_Emb20/',
        "label": 'POD',
        "q_inf": 20
    }]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    args = parser.parse_args()
    working_path = args.working_path+'/'

    recons_s_ref = np.load('datasets_rubber_hyperelastic_cantilever_big_range/FOM/FOM_300steps_random_seed4.npy')
    nmrom_s_ref = recons_s_ref

    recons_r_ref = np.load('datasets_rubber_hyperelastic_cantilever_big_range/R_test_small.npy')
    nmrom_r_ref = recons_r_ref

    model_info_list=[]
    for case in result_cases:
        model_info_list.append(Model_Info(working_path, case["model_path"], case["label"], case["q_inf"], recons_s_ref, nmrom_s_ref, recons_r_ref, nmrom_r_ref))


    plot_s_error(model_info_list)
    plot_rNoRorce_error(model_info_list)
    plot_rRorce_error(model_info_list)


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



