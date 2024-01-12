# from nn_trainer import NN_Trainer
from nn_trainer import NN_Trainer
from sys import argv
import json

# Import pycompss
""" from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import *
from pycompss.api.constraint import constraint """

""" @constraint(computing_units="8")"""
""" @task() """
def train(working_path, ae_config):
    training_routine=NN_Trainer(working_path, ae_config)
    training_routine.execute_training()
    del training_routine

def prepare_files(orig_project_parameters_file, dataset_path, working_path):
    """pre-pending the absolut path of the files in the Project Parameters"""
    output_path=orig_project_parameters_file[:-5]+'_workflow.json'

    with open(working_path+dataset_path+orig_project_parameters_file,'r') as f:
        updated_project_parameters = json.load(f)
        file_input_name = updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"]
        materials_filename = updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"]
        updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = working_path + file_input_name
        updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"] = working_path + materials_filename

    with open(working_path+dataset_path+output_path,'w') as f:
        json.dump(updated_project_parameters, f, indent = 4)
    
    return output_path

if __name__ == "__main__":

    config_list_examples = [
   {
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'PODANN', # ['POD','Quad','PODANN]
            "q_inf_size": 6,
            "q_sup_size": 20,
            "hidden_layers": [40,40],
            "prepost_process": 'svd_white_nostand',
            "opt_strategy": {
                "name": 'tf_sonly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed']
                "r_loss_type": 'norm',  # ['norm, 'diff']
                "r_loss_log_scale": True,
                "learning_rate": ('steps', 1e-4, 10, 1e-6, 500), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250, 2), ('sgdr', 0.001, 1e-6, 250, 2)
                "batch_size": 1,
                "epochs": 1000,
                # "wx": ('const', 0.001),
                # "wr": ('const', 0.001),
            },
            # "finetune_from": 'saved_models/CorrDecoder_RNormLoss_Dense_extended_Augmented_SVDWhiteNonStand_emb6_lay40_LRsteps_1000ep/',
            "finetune_from": None,
            "augmented": True,
            "use_bias": False,
        },
        "dataset_path": 'datasets_two_forces_dense_extended/',
        "models_path_root": 'saved_models/',
        "project_parameters_file":'ProjectParameters_tf.json'
   },
   {
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'Quad', # ['POD','Quad','PODANN]
            "q_size": 6,
            "prepost_process": 'scale_global', # ['scale_global', 'identity']
            "opt_strategy": {
                "name": 'tf_sonly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'least_squares']
                "r_loss_type": 'norm',  # ['norm, 'diff']
                "r_loss_log_scale": True,
                # "learning_rate": ('steps', 1e-4, 10, 1e-6, 500), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
                # "batch_size": 1,
                # "epochs": 1000,
                # "wx": ('const', 0.001),
                # "wr": ('const', 0.001),
                # "add_init_noise": False
            },
            "finetune_from": '',
            "augmented": False,
        },
        "dataset_path": 'datasets_two_forces_dense_extended/',
        "models_path_root": 'saved_models/',
        "project_parameters_file":'ProjectParameters_tf.json'
   },
   {
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'POD', # ['POD','Quad','PODANN]
            "q_size": 6,
            "augmented": False,
            "opt_strategy": {
                "r_loss_type": 'norm',  # ['norm, 'diff']
                "r_loss_log_scale": True
            },
            "finetune_from": '',
        },
        "dataset_path": 'datasets_two_forces_dense_extended/',
        "models_path_root": 'saved_models/',
        "project_parameters_file":'ProjectParameters_tf.json'
   },
   ]
    
    train_configs_list = [
#     {
#         "sim_type": 'fluid',
#         "name": None,
#         "architecture": {
#             "name": 'POD', # ['POD','Quad','PODANN]
#             "q_size": 200,
#             "augmented": False,
#             "opt_strategy": {
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False
#             },
#             "finetune_from": None,
#         },
#         "dataset_path": 'datasets_fluid_past_cylinder/',
#         "models_path_root": 'saved_models/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    }
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'Quad', # ['POD','Quad','PODANN]
#             "q_size": 6,
#             "prepost_process": 'scale_global', # ['scale_global', 'identity']
#             "opt_strategy": {
#                 "name": 'least_squares', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'least_squares']
#                 "r_loss_type": 'norm',  # ['norm, 'diff']
#                 "r_loss_log_scale": True,
#                 "learning_rate": ('const', 1e-6), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#                 # "batch_size": 1,
#                 # "epochs": 1,
#                 # "wx": ('const', 0.001),
#                 # "wr": ('const', 0.001),
#                 "add_init_noise": False
#             },
#             "finetune_from": None,
#             "augmented": False,
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever/',
#         "models_path_root": 'saved_models/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'POD', # ['POD','Quad','PODANN]
#             "q_size": 6,
#             "augmented": False,
#             "opt_strategy": {
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False
#             },
#             "finetune_from": None,
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
#         "models_path_root": 'saved_models_cantilever_big_range/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    }
#    {
#         "sim_type": 'fluid',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 20,
#             "q_sup_size": 200,
#             "hidden_layers": [400,400],
#             "prepost_process": 'svd_white_nostand_crop',
#             "opt_strategy": {
#                 "name": 'tf_sonly_cropped', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 500, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 2000
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": 0.4,
#         },
#         "dataset_path": 'datasets_fluid_past_cylinder_bdf2/',
#         "models_path_root": 'saved_models_fluid_bdf2/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },{
    {
        "sim_type": 'structural',
        "name": None,
        "architecture": {
            "name": 'POD', # ['POD','Quad','PODANN]
            "q_size": 40,
            "opt_strategy": {
                "r_loss_type": 'diff',  # ['norm, 'diff']
                "r_loss_log_scale": False,
            },
            "finetune_from": None,
            "augmented": False
        },
        "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
        "models_path_root": 'saved_models_cantilever_big_range/',
        "project_parameters_file":'ProjectParameters_tf.json'
   },
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 20,
#             "q_sup_size": 60,
#             "hidden_layers": [200,200],
#             "prepost_process": 'svd_white_nostand',
#             "opt_strategy": {
#                 "name": 'tf_ronly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.0001, 1e-6, 400, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 800
#             },
#             "finetune_from": 'saved_models_cantilever_big_range/PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001_slower/',
#             # "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
#         "models_path_root": 'saved_models_cantilever_big_range/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 14,
#             "q_sup_size": 60,
#             "hidden_layers": [200,200],
#             "prepost_process": 'svd_rerange',
#             "opt_strategy": {
#                 "name": 'tf_sonly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 500, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 1000
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": True,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
#         "models_path_root": 'saved_models_cantilever_big_range/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 6,
#             "q_sup_size": 60,
#             "hidden_layers": [200,200],
#             "prepost_process": 'svd',
#             "opt_strategy": {
#                 "name": 'tf_sfarhat', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 200, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 256,
#                 "epochs": 800
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
#         "models_path_root": 'saved_models_cantilever_big_range/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'structural',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 14,
#             "q_sup_size": 60,
#             "hidden_layers": [200,200],
#             "prepost_process": 'svd_white_nostand',
#             "opt_strategy": {
#                 "name": 'tf_sonly', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 500, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 1000
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_rubber_hyperelastic_cantilever_big_range/',
#         "models_path_root": 'saved_models_cantilever_big_range/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'fluid',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 20,
#             "q_sup_size": 200,
#             "hidden_layers": [400,400],
#             "prepost_process": 'svd_white_nostand_crop',
#             "opt_strategy": {
#                 "name": 'tf_sonly_cropped', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 200, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 800
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_fluid_past_cylinder_bdf2/',
#         "models_path_root": 'saved_models_fluid_bdf2/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    },
#    {
#         "sim_type": 'fluid',
#         "name": None,
#         "architecture": {
#             "name": 'PODANN', # ['POD','Quad','PODANN]
#             "q_inf_size": 20,
#             "q_sup_size": 200,
#             "hidden_layers": [400,400],
#             "prepost_process": 'svd_white_nostand_crop',
#             "opt_strategy": {
#                 "name": 'tf_srmixed_cropped', # ['tf_sonly', 'tf_ronly', 'tf_srmixed', 'tf_wonly']
#                 "r_loss_type": 'diff',  # ['norm, 'diff']
#                 "r_loss_log_scale": False,
#                 "learning_rate": ('sgdr', 0.001, 1e-6, 200, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
#                 "batch_size": 16,
#                 "epochs": 800,
#                 "wx": ('const', 0.01),
#                 "wr": ('const', 0.99)
#             },
#             # "finetune_from": 'saved_models/PODANN/PODANN_tf_srmixed_diff_svd_white_nostand_Lay[40, 40]_Emb6.40_LRsteps0.001/',
#             "finetune_from": None,
#             "augmented": False,
#             "use_bias": False,
#             "use_dropout": None
#         },
#         "dataset_path": 'datasets_fluid_past_cylinder_bdf2/',
#         "models_path_root": 'saved_models_fluid_bdf2/',
#         "project_parameters_file":'ProjectParameters_tf.json'
#    }
   ]

    working_path=argv[1]+"/"
    
    for i, train_config in enumerate(train_configs_list):
        
        print('----------  Training case ', i+1, ' of ', len(train_configs_list), '  ----------')
        output_path=prepare_files(train_config["project_parameters_file"], train_config["dataset_path"], working_path)
        train_config["project_parameters_file"]=output_path
        train(working_path, train_config)
    
    # compss_barrier()
    print('FINISHED TRAINING')

        
