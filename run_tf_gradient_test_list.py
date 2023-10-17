from sys import argv
import argparse

from tf_gradient_tester import TF_Gradient_Tester

def test_gradients(working_path, model_path):
    test_routine=TF_Gradient_Tester(working_path, model_path)
    test_routine.execute_test()

if __name__ == "__main__":
    paths_list=[
            # 'Quad/Quad_least_squares_identity_Emb6',
            'PODANN/PODANN_tf_wonly_vector_diff_svd_white_nostand_Lay[40, 40]_Emb6.20_LRsteps0.001',
            ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        test_gradients(working_path, 'saved_models/'+model_path+'/')
    
    # compss_barrier()
    print('FINISHED EVALUATING')