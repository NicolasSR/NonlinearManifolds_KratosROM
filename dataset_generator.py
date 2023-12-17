import KratosMultiphysics
from KratosMultiphysics.RomApplication.rom_manager import RomManager
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import numpy as np


from scipy.stats import qmc

# import locale
# locale.setlocale(locale.LC_ALL, 'en_US.utf8')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def CustomizeSimulation(cls, global_model, parameters):

    class CustomSimulation(cls):

        def __init__(self, model,project_parameters, custom_param = None):
            super().__init__(model,project_parameters)
            self.custom_param  = custom_param
            """
            Customize as needed
            """

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()

            self.snapshots_matrix = list(np.load("FOM.npy"))
            #self.residuals_matrix = list(np.load("FOM_RESIDUALS.npy"))
            self.pointload_matrix = list(np.load("FOM_POINTLOADS.npy"))

            #self.fixedNodes = []
            self.main_model_part = self.model.GetModelPart("Structure")

        def Initialize(self):
            super().Initialize()
            """
            Customize as needed
            """

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            """ super().FinalizeSolutionStep()

            strategy  = self._GetSolver()._GetSolutionStrategy()
            buildsol  = self._GetSolver()._GetBuilderAndSolver()
            scheme    = KratosMultiphysics.ResidualBasedIncrementalUpdateStaticScheme()

            A = strategy.GetSystemMatrix()
            b = strategy.GetSystemVector()

            space = KratosMultiphysics.UblasSparseSpace()

            space.SetToZeroMatrix(A)
            space.SetToZeroVector(b)

            buildsol.Build(scheme, self.main_model_part, A, b)

            self.residuals_matrix.append([x for x in b])
            """

        def Finalize(self):
            super().Finalize()
            snapshot = []
            for node in self.main_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)
            pointload = []
            for condition in self.main_model_part.Conditions:
                pointload.append(condition.GetValue(SMA.LINE_LOAD))
            self.pointload_matrix.append(pointload)

            np.save("FOM.npy",           self.snapshots_matrix)
            #np.save("FOM_RESIDUALS.npy", self.residuals_matrix)
            np.save("FOM_POINTLOADS.npy", self.pointload_matrix)

            #self.testerro = np.load("FOM.npy")

            #print(self.snapshots_matrix - self.testerro)


        def CustomMethod(self):
            """
            Customize as needed
            """
            return self.custom_param

    return CustomSimulation(global_model, parameters)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def UpdateProjectParameters(parameters, mu=None):
    """
    Customize ProjectParameters here for imposing different conditions to the simulations as needed
    """
    parameters["problem_data"]["end_time"].SetDouble(mu[2])
    parameters["processes"]["loads_process_list"][0]["Parameters"]["modulus"].SetString(str(mu[0]/mu[2])+"*t")
    parameters["processes"]["loads_process_list"][1]["Parameters"]["modulus"].SetString(str(mu[1]/mu[2])+"*t")

    return parameters
    
def UpdateMaterialParametersFile(parameters, mu=None):
    return parameters


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def GetRomManagerParameters():
    """
    This function allows to easily modify all the parameters for the ROM simulation.
    The returned KratosParameter object is seamlessly used inside the RomManager.
    """
    general_rom_manager_parameters = KratosMultiphysics.Parameters("""{
            "rom_stages_to_train" : [],             // ["ROM","HROM"]
                "rom_stages_to_test" : [],              // ["ROM","HROM"]
                "paralellism" : null,                        // null, TODO: add "compss"
                "projection_strategy": "galerkin",            // "lspg", "galerkin", "petrov_galerkin"
                "assembling_strategy": "global",            // "global", "elemental"
                "save_gid_output": false,                    // false, true #if true, it must exits previously in the ProjectParameters.json
                "save_vtk_output": false,                    // false, true #if true, it must exits previously in the ProjectParameters.json
                "output_name": "id",                         // "id" , "mu"
                "ROM":{
                    "svd_truncation_tolerance": 1e-6,
                    "model_part_name": "Structure",                            // This changes depending on the simulation: Structure, FluidModelPart, ThermalPart #TODO: Idenfity it automatically
                    "nodal_unknowns": ["DISPLACEMENT_X","DISPLACEMENT_Y"],     // Main unknowns. Snapshots are taken from these
                    "rom_basis_output_format": "numpy",                         
                    "rom_basis_output_name": "RomParameters",
                    "snapshots_control_type": "step",                          // "step", "time"
                    "snapshots_interval": 1,
                    "galerkin_rom_bns_settings": {
                        "monotonicity_preserving": false
                    },
                    "lspg_rom_bns_settings": {
                        "train_petrov_galerkin": false,             
                        "basis_strategy": "residuals",                        // 'residuals', 'jacobian'
                        "include_phi": false,
                        "svd_truncation_tolerance": 0.001,
                        "solving_technique": "normal_equations",              // 'normal_equations', 'qr_decomposition'
                        "monotonicity_preserving": false
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":


    def random_samples_from_interval(initial, final, number_of_samples):
        import numpy as np
        return initial + np.random.rand(number_of_samples)*(final-initial)

    def get_multiple_params():

        sampler_test = qmc.Halton(d=2, seed=4)

        mu=sampler_test.random(n=300)
        mu=qmc.scale(mu, [-3000,-3000], [3000, 3000])

        mu_steps=np.expand_dims(np.linalg.norm(mu, axis=1)//10,axis=1)
        mu=np.concatenate([mu,mu_steps], axis=1)
        return mu[278:]
    
    dataset_path='datasets_rubber_hyperelastic_cantilever_big_range/'
    
    snapshots_matrix = []
    #residuals_matrix = []
    pointload_matrix = []

    np.save("FOM.npy",           snapshots_matrix)
    #np.save("FOM_RESIDUALS.npy", residuals_matrix)
    np.save("FOM_POINTLOADS.npy", pointload_matrix)


    mu = get_multiple_params() # random train parameters

    #mu_train =  [[param1, param2, ..., param_p]] #list of lists containing values of the parameters to use in POD
    general_rom_manager_parameters = GetRomManagerParameters()
    project_parameters_name = dataset_path+"ProjectParameters_FOM.json"

    rom_manager = RomManager(project_parameters_name,general_rom_manager_parameters,CustomizeSimulation,UpdateProjectParameters, UpdateMaterialParametersFile)

    """if no list "mu" is passed, the case already contained in the ProjectParametes and CustomSimulation is launched (useful for example for a single time dependent simulation)"""
    # rom_manager.Fit(mu, store_all_snapshots=False)#, store_residuals=True)
    rom_manager.RunFOM(mu)#, store_residuals=True)
    rom_manager.PrintErrors()

