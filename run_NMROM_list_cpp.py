from sys import argv
import argparse

from NMROM_simulator_sparse_cpp import NMROM_Simulator
from NMROM_simulator_sparse_cpp_pod import NMROM_POD_Simulator

def simulate(working_path, sim_config):
    if sim_config["projection_strategy"] == 'custom':
        simulation_routine=NMROM_Simulator(working_path, sim_config, sim_config["best"])
    elif sim_config["projection_strategy"] == 'pod':
        simulation_routine=NMROM_POD_Simulator(working_path, sim_config)
    else:
        simulation_routine=None
    simulation_routine.execute_simulation()

if __name__ == "__main__":
    sim_configs_list=[
#    {
#         "model_path": 'saved_models_cantilever_big_range/PODANN/PODANN_tf_ronly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001_slower',
#         "projection_strategy": 'custom', # ['custom', 'pod']
#         "parameters_selection_strategy": 'random', # ['progressive', 'random']
#         "best": 'r'
#    }
   {
        "model_path": 'saved_models_cantilever_big_range/POD/POD_Emb18',
        "projection_strategy": 'pod', # ['custom', 'pod']
        "parameters_selection_strategy": 'random', # ['progressive', 'random']
   }
   ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    # parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    # best=args.best
    
    for i, sim_config in enumerate(sim_configs_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(sim_configs_list), '  ----------')
        simulate(working_path, sim_config)
    
    # compss_barrier()
    print('FINISHED EVALUATING')