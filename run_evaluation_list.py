from sys import argv
import argparse

from nn_evaluator import NN_Evaluator

def evaluate(working_path, model_path, GID_FOM_filename, best, test_validation=False):
    evaluation_routine=NN_Evaluator(working_path, model_path, GID_FOM_filename, best, test_validation=test_validation)
    evaluation_routine.execute_evaluation()

if __name__ == "__main__":
    paths_list=[
            # 'Quad/Quad_least_squares_identity_Emb6',
            'PODANN/PODANN_tf_ronly_diff_noLog_svd_white_nostand_Lay[40, 40]_Emb6.20_LRtri20.001_lrscale10',
            ]
    
    GID_FOM_filename='FOM_equalForces_300steps.npy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    parser.add_argument('--test_val', action='store_true')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    test_validation = args.test_val
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        evaluate(working_path, 'saved_models/'+model_path+'/', GID_FOM_filename, best, test_validation=test_validation)
    
    # compss_barrier()
    print('FINISHED EVALUATING')