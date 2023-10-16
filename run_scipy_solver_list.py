from sys import argv
import argparse

from scipy_solver import Scipy_Solver

def solve(working_path, model_path, best):
    evaluation_routine=Scipy_Solver(working_path, model_path, best)
    evaluation_routine.execute_solver()

if __name__ == "__main__":
    paths_list=[
            'Quad/Quad_least_squares_identity_Emb6',
            ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        solve(working_path, 'saved_models/'+model_path+'/', best)
    
    print('FINISHED SIMULATING')