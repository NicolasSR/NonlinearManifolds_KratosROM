import numpy as np

import os

import KratosMultiphysics
from KratosMultiphysics.RomApplication.rom_manager import RomManager

from Kratos_Simulators.structural_mechanics_kratos_simulator import StructuralMechanics_KratosSimulator

from ArchitectureFactories.PODANN_factory import PODANN_Architecture_Factory
from ArchitectureFactories.Quad_factory import Quad_Architecture_Factory
from ArchitectureFactories.POD_factory import POD_Architecture_Factory

import tensorflow as tf

def CustomizeSimulation(cls, global_model, parameters, nn_rom_interface=None):

    class CustomSimulation(cls):

        def __init__(self, model,project_parameters, custom_param = None, nn_rom_interface=None):
            super().__init__(model,project_parameters)
            self.custom_param  = custom_param
            self.nn_rom_interface = nn_rom_interface
            self.snapshots_matrix_filename=nn_rom_interface.get_snapshots_matrix_filename()

            self._EncodeSnapshot=nn_rom_interface.get_encode_function()
            self._DecodeSnapshot=nn_rom_interface.get_decode_function()
            self._GetDecoderGradient=nn_rom_interface.get_get_decoder_gradient_function()

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()
            self.snapshots_matrix = list(np.load(self.snapshots_matrix_filename))
            self.computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()
            
        def Initialize(self):
            super().Initialize()
            self._ConfigureEncoderDecoder()
            """
            Customize as needed
            """
            
        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            snapshot = []
            for node in self.computing_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)
            
        def Finalize(self):
            super().Finalize()
            np.save(self.snapshots_matrix_filename, self.snapshots_matrix)

        def CustomMethod(self):
            """
            Customize as needed
            """
            return self.custom_param

    return CustomSimulation(global_model, parameters, nn_rom_interface=nn_rom_interface)


class NN_ROM_Interface():
    def __init__(self, arch_factory, prepost_processor, enc_network, dec_network, results_path):
        self.arch_factory=arch_factory
        self.prepost_processor=prepost_processor
        self.enc_network=enc_network
        self.dec_network=dec_network

        self.snapshots_matrix_filename=results_path+'ROM_snaps.npy'


    def get_encode_function(self):
        return self.arch_factory.NMROM_encoder(self.prepost_processor, self.enc_network)
    
    def get_decode_function(self):
        return self.arch_factory.NMROM_decoder(self.prepost_processor, self.dec_network)
    
    def get_get_decoder_gradient_function(self):
        return self.arch_factory.NMROM_decoder_gradient(self.prepost_processor, self.dec_network)
    
    def get_snapshots_matrix_filename(self):
        return self.snapshots_matrix_filename
    

class NMROM_Simulator():

    def __init__(self, working_path, sim_config, best):

        self.sim_config=sim_config

        self.working_path=working_path
        self.model_path=working_path+'saved_models/'+self.sim_config["model_path"]+'/'
        self.results_path=self.model_path+'NMROM_simulation_results/'
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx_'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr_'
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
            self.best_name_part=''
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        self.GID_FOM_filename = 'NMROM_GID_out'

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        with open(self.model_path+"train_config.npy", "rb") as train_config_file:
            self.train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        print(self.train_config)
        self.dataset_path=working_path+self.train_config['dataset_path']

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename
    
    def architecture_factory_selector(self, arch_config):
        arch_type = arch_config["name"]
        if arch_type == 'PODANN':
            return PODANN_Architecture_Factory(self.working_path, arch_config)
        elif arch_type == 'Quad':
            return Quad_Architecture_Factory(self.working_path, arch_config)
        elif arch_type == 'POD': 
            return POD_Architecture_Factory(self.working_path, arch_config)
        else:
            print('No valid architecture type was selected')
            return None
        
    def kratos_simulator_selector(self, sim_type):
        # if 'fluid' in sim_type:
        #     return KratosSimulator_Fluid
        # else:
        return StructuralMechanics_KratosSimulator
    
    def get_orig_fom_snapshots(self):
        S_FOM_orig=np.load(self.working_path+self.train_config['dataset_path']+'FOM.npy')
        return S_FOM_orig

    def  get_update_strategy_and_parameters_filename(self, parameters_selection_strategy):

        def random_samples_from_interval(initial, final, number_of_samples):
            return initial + np.random.rand(number_of_samples)*(final-initial)

        def get_multiple_params():
            print('No prior force values were found. Creating new ones and saving them')
            number_of_params = 300
            f_modulus0 = random_samples_from_interval(0.0,6.0e7,number_of_params)
            f_modulus1 = random_samples_from_interval(0.0,6.0e7,number_of_params)

            mu = []
            for i in range(number_of_params):
                mu.append([f_modulus0[i],f_modulus1[i]])
            return mu
        
        def UpdateProjectParametersNone(parameters, mu=None):
            """
            Customize ProjectParameters here for imposing different conditions to the simulations as needed
            """
            return parameters

        def UpdateProjectParametersRandom(parameters, mu=None):
            """
            Customize ProjectParameters here for imposing different conditions to the simulations as needed
            """
            parameters["processes"]["loads_process_list"][0]["Parameters"]["modulus"].SetString(str(mu[0]))
            parameters["processes"]["loads_process_list"][1]["Parameters"]["modulus"].SetString(str(mu[1]))

            return parameters

        if parameters_selection_strategy == "progressive":
            print('Using progressive parameters strategy')
            UpdateProjectParameters = UpdateProjectParametersNone
            # project_parameters_name = "ProjectParameters_recons.json"
            project_parameters_name = "ProjectParameters_recons_3000steps.json"
            mu = [None]
        elif parameters_selection_strategy == "random":
            print('Using random parameters strategy')
            UpdateProjectParameters =  UpdateProjectParametersRandom
            project_parameters_name = "ProjectParameters_tf.json"
            mu = get_multiple_params()
        else:
            print('Parameters selection strategy not valid')
            UpdateProjectParameters =  None
            project_parameters_name = None
            mu = None

        project_parameters_name=self.dataset_path+project_parameters_name

        return project_parameters_name, UpdateProjectParameters, mu

    def GetRomManagerParameters(self, projection_strategy):
        """
        This function allows to easily modify all the parameters for the ROM simulation.
        The returned KratosParameter object is seamlessly used inside the RomManager.
        """
        general_rom_manager_parameters = KratosMultiphysics.Parameters("""{
                "rom_stages_to_train" : [],             // ["ROM","HROM"]
                "rom_stages_to_test" : [],              // ["ROM","HROM"]
                "paralellism" : null,                        // null, TODO: add "compss"
                "projection_strategy": """+'"'+projection_strategy+'"'+""",                  // "lspg", "galerkin", "petrov_galerkin", "custom", "custom_lspg"
                "save_gid_output": true,                    // false, true #if true, it must exits previously in the ProjectParameters.json
                "save_vtk_output": false,                    // false, true #if true, it must exits previously in the ProjectParameters.json
                "output_name": "id",                         // "id" , "mu"
                "ROM":{
                    "svd_truncation_tolerance": 1e-6,
                    "model_part_name": "Structure",                            // This changes depending on the simulation: Structure, FluidModelPart, ThermalPart #TODO: Idenfity it automatically
                    "nodal_unknowns": ["DISPLACEMENT_X","DISPLACEMENT_Y"],     // Main unknowns. Snapshots are taken from these
                    "rom_basis_output_format": "json",                         //TODO: add "numpy"
                    "rom_basis_output_name": "RomParameters",
                    "snapshots_control_type": "step",                          // "step", "time"
                    "snapshots_interval": 1,
                    "petrov_galerkin_training_parameters":{
                        "basis_strategy": "residuals",                         // 'residuals', 'jacobian'
                        "include_phi": false,
                        "svd_truncation_tolerance": 0,
                        "echo_level": 0
                    }
                },
                "HROM":{
                    "element_selection_type": "empirical_cubature",
                    "element_selection_svd_truncation_tolerance": 0,
                    "create_hrom_visualization_model_part" : true,
                    "echo_level" : 0
                }
            }""")

        return general_rom_manager_parameters

    def execute_simulation(self):

        # Select the architecture to use
        arch_factory = self.architecture_factory_selector(self.train_config["architecture"])

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation_class=self.kratos_simulator_selector(self.train_config["sim_type"])
        self.kratos_simulation = kratos_simulation_class(self.working_path, self.train_config)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        # Select the type of preprocessing (normalisation)
        prepost_processor=arch_factory.prepost_processor_selector(self.working_path, self.train_config["dataset_path"])

        S_FOM_orig = self.get_orig_fom_snapshots()
        arch_factory.configure_prepost_processor(prepost_processor, S_FOM_orig, crop_mat_tf, crop_mat_scp)

        print('======= Instantiating TF Model =======')
        network, enc_network, dec_network = arch_factory.define_network(prepost_processor, self.kratos_simulation)
        
        print('======= Loading TF Model weights =======')
        network.load_weights(self.model_weights_path+self.model_weights_filename)

        self.kratos_simulation=None

        print('======= Setting up NM ROM simulation =======')

        general_rom_manager_parameters = self.GetRomManagerParameters(self.sim_config["projection_strategy"])
        
        project_parameters_name, UpdateProjectParameters, mu = self.get_update_strategy_and_parameters_filename(self.sim_config["parameters_selection_strategy"])

        nn_rom_interface = NN_ROM_Interface(arch_factory, prepost_processor, enc_network, dec_network, self.results_path)

        def UpdateMaterialParametersFile(parameters, mu=None):
            return parameters
        
        rom_manager = RomManager(project_parameters_name,general_rom_manager_parameters,CustomizeSimulation,UpdateProjectParameters,UpdateMaterialParametersFile)
        
        print('======= Running NM ROM simulation =======')
        snapshots_matrix = []
        np.save(nn_rom_interface.get_snapshots_matrix_filename(), snapshots_matrix)
        rom_manager.RunROM(mu, nn_rom_interface=nn_rom_interface, output_path=self.results_path)
        rom_manager.PrintErrors()

