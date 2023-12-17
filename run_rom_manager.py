import KratosMultiphysics
from KratosMultiphysics.RomApplication.rom_manager import RomManager
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import numpy as np

# import locale
# locale.setlocale(locale.LC_ALL, 'en_US.utf8')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def CustomizeSimulation(cls, global_model, parameters):

    class CustomSimulation(cls):

        def __init__(self, model,project_parameters, custom_param = None):
            super().__init__(model,project_parameters)
            self.custom_param  = custom_param

            self.mu = get_multiple_params()
            self.mu_counter = 0
            self.project_parameters["problem_data"]["end_time"].SetInt(len(self.mu))
            """
            Customize as needed
            """

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()

            #self.snapshots_matrix = list(np.load("FOM.npy"))
            #self.residuals_matrix = list(np.load("FOM_RESIDUALS.npy"))
            self.pointload_matrix = list(np.load("FOM_POINTLOADS.npy"))

            #self.fixedNodes = []
            self.main_model_part = self.model.GetModelPart("Structure")
            self.point_load_h_model_part = self.model.GetModelPart("Structure.LineLoad2D_ForceH")
            self.point_load_v_model_part = self.model.GetModelPart("Structure.LineLoad2D_ForceV")

        def Initialize(self):
            super().Initialize()
            """
            Customize as needed
            """

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
            point_load_h = KratosMultiphysics.Vector(np.array([-self.mu[self.mu_counter][0],0.0 ,0.0]))
            for condition in self.point_load_h_model_part.Conditions:
                condition.SetValue(SMA.LINE_LOAD, point_load_h)
            point_load_v = KratosMultiphysics.Vector(np.array([0.0, -self.mu[self.mu_counter][1] ,0.0]))
            for condition in self.point_load_v_model_part.Conditions:
                condition.SetValue(SMA.LINE_LOAD, point_load_v)
            self.mu_counter += 1

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            pointload = []
            for condition in self.main_model_part.Conditions:
                pointload.append(condition.GetValue(SMA.LINE_LOAD))
            self.pointload_matrix.append(pointload)


        def Finalize(self):
            super().Finalize()
            np.save("FOM_POINTLOADS.npy", self.pointload_matrix)


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
    # parameters["processes"]["loads_process_list"][0]["Parameters"]["modulus"].SetString(str(mu[0]))
    # parameters["processes"]["loads_process_list"][1]["Parameters"]["modulus"].SetString(str(mu[1]))

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
            "rom_stages_to_train" : ["ROM"],             // ["ROM","HROM"]
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
        number_of_params = 10000
        f_modulus0 = random_samples_from_interval(0.0,600,number_of_params)
        f_modulus1 = random_samples_from_interval(0.0,600,number_of_params)
        mu = []
        for i in range(number_of_params):
            mu.append([f_modulus0[i],f_modulus1[i]])
        return mu
    
    pointload_matrix = []

    np.save("FOM_POINTLOADS.npy", pointload_matrix)

    general_rom_manager_parameters = GetRomManagerParameters()
    project_parameters_name = "datasets_two_forces_dense_extended/ProjectParameters_dataset_gen.json"

    rom_manager = RomManager(project_parameters_name,general_rom_manager_parameters,CustomizeSimulation,UpdateProjectParameters, UpdateMaterialParametersFile)

    """if no list "mu" is passed, the case already contained in the ProjectParametes and CustomSimulation is launched (useful for example for a single time dependent simulation)"""
    rom_manager.Fit(store_all_snapshots=True)#, store_residuals=True)
    rom_manager.PrintErrors()


