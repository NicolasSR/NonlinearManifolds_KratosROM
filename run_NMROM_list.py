from sys import argv
import argparse

from NMROM_simulator import NMROM_Simulator

def simulate(working_path, sim_config, best):
    simulation_routine=NMROM_Simulator(working_path, sim_config, best)
    simulation_routine.execute_simulation()

if __name__ == "__main__":
    sim_configs_list=[
   {
        # "model_path": 'Quad/Quad_least_squares_identity_Emb6',
        "model_path": 'PODANN/PODANN_tf_ronly_diff_noLog_svd_white_nostand_Lay[40, 40]_Emb6.20_LRtri20.001_lrscale10',
        "projection_strategy": 'custom', # ['custom', 'custom_lspg']
        "parameters_selection_strategy": 'progressive', # ['progressive', 'random']
   },
   ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    
    for i, sim_config in enumerate(sim_configs_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(sim_configs_list), '  ----------')
        simulate(working_path, sim_config, best)
    
    # compss_barrier()
    print('FINISHED EVALUATING')