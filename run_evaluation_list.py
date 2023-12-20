from sys import argv
import argparse

from nn_evaluator import NN_Evaluator

def evaluate(working_path, model_path, GID_FOM_filename, best, test_validation=False, test_small=False):
    evaluation_routine=NN_Evaluator(working_path, model_path, GID_FOM_filename, best, test_validation=test_validation, test_small=test_small)
    evaluation_routine.execute_evaluation()

if __name__ == "__main__":
    paths_list=[
            # 'POD/POD_Emb15'
            # 'Quad/Quad_least_squares_scale_global_Emb6',
            # 'saved_models_cantilever_big_range/PODANN_Standalone/test_sonly_lrsgdr'
            # 'saved_models_cantilever_big_range/PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.001'
            'saved_models_cantilever_big_range/POD/POD_Emb20'
            # 'saved_models_fluid_bdf2/PODANN/PODANN_tf_sonly_cropped_diff_svd_white_nostand_crop_Lay[400, 400]_Emb20.200_LRsgdr0.001',
            # 'saved_models/PODANN/s_loss_bis',
            ]
    
    # GID_FOM_filename='FOM_equalForces_500steps.npy'
    GID_FOM_filename='FOM_300steps_random_seed4.npy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    parser.add_argument('--test_val', action='store_true')
    parser.add_argument('--test_small', action='store_true')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    test_validation = args.test_val
    test_small = args.test_small
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        evaluate(working_path, model_path+'/', GID_FOM_filename, best, test_validation=test_validation, test_small=test_small)
    
    # compss_barrier()
    print('FINISHED EVALUATING')