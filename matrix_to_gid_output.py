import numpy as np
import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.gid_output_process as GOP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

def create_out_mdpa(model_part, file_name):
    model_part.AddNodalSolutionStepVariable(KMP.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KMP.REACTION)

    import_flags = KMP.ModelPartIO.READ

    KMP.ModelPartIO(file_name, import_flags).ReadModelPart(model_part)

def print_results_to_gid(model_part, snapshot_matrix, residuals_matrix):

    gid_output = GOP.GiDOutputProcess(
        model_part,
        "PredictDiff",
        KMP.Parameters("""
            {
                "result_file_configuration" : {
                    "gidpost_flags"               : {
                        "GiDPostMode"           : "GiD_PostAscii",
                        "WriteDeformedMeshFlag" : "WriteDeformed",
                        "WriteConditionsFlag"   : "WriteConditions",
                        "MultiFileFlag"         : "SingleFile"
                    },
                    "file_label"                  : "time",
                    "output_control_type"         : "step",
                    "output_interval"             : 1,
                    "body_output"                 : true,
                    "node_output"                 : false,
                    "skin_output"                 : false,
                    "plane_output"                : [],
                    "nodal_results"               : ["DISPLACEMENT","REACTION"],
                    "gauss_point_results"         : [],
                    "nodal_nonhistorical_results" : []
                },
                "point_data_configuration"  : []
            }"""
        )
    )

    gid_output.ExecuteInitialize()

    print(snapshot_matrix.shape)
    for ts in range(0, snapshot_matrix.shape[0]):
    # for ts in range(1):
    # ts=snapshot_matrix.shape[0]-1
    # for k in range(1):
        model_part.ProcessInfo[KMP.STEP] = ts+1
        model_part.ProcessInfo[KMP.TIME] = ts+1
        gid_output.ExecuteBeforeSolutionLoop()
        gid_output.ExecuteInitializeSolutionStep()

        snapshot = snapshot_matrix[ts]
        residuals = residuals_matrix[ts]

        # var_utils.SetSolutionStepValuesVector(model_part.Nodes, KMP.DISPLACEMENT, snapshot, 2)
        # var_utils.SetSolutionStepValuesVector(model_part.Nodes, KMP.REACTION, residuals, 2)
        
        i=0
        c=2
        for node in model_part.Nodes:
            node.SetSolutionStepValue(KMP.DISPLACEMENT_X,0,snapshot[i*c+0])
            node.SetSolutionStepValue(KMP.DISPLACEMENT_Y,0,snapshot[i*c+1])
            node.SetSolutionStepValue(KMP.REACTION_X,0,residuals[i*c+0])
            node.SetSolutionStepValue(KMP.REACTION_Y,0,residuals[i*c+1])
            i+=1

        # conditions_array=model_part.Conditions
        # for i, condition in enumerate(conditions_array):
        #     condition.SetValue(SMA.LINE_LOAD, forces)

        gid_output.PrintOutput()
        gid_output.ExecuteFinalizeSolutionStep()

    gid_output.ExecuteFinalize()


if __name__ == "__main__":

    # snapshots_matrix=np.load('saved_models_cantilever_big_range/POD/POD_Emb20/NMROM_simulation_results_random300/ROM_snaps_converged_corrected.npy')
    snapshots_matrix=np.load('saved_models_cantilever_big_range/PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb12.60_LRsgdr0.001/NMROM_simulation_results_random300/ROM_snaps_converged_corrected.npy')
    # snapshots_matrix=np.load('Quad_x_snapshots.npy')
    # snapshots_matrix=np.load('Rel_quad_scipy_diff.npy')
    # snapshots_matrix=np.load('FOM_snaps_30steps.npy')
    reactions_matrix=np.load('saved_models_cantilever_big_range/PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb12.60_LRsgdr0.001/NMROM_simulation_results_random300/ROM_residuals_converged_corrected.npy')
    # reactions_matrix=np.load('Quad_reactions_snapshots.npy')

    # reactions_matrix_log=np.log(1+np.abs(reactions_matrix))
    # reactions_matrix_sign=reactions_matrix/np.abs(reactions_matrix)
    # reactions_matrix=reactions_matrix_log*reactions_matrix_sign

    # plt.plot(reactions_matrix[0][0])
    # plt.show()
    # exit()

    # reactions_matrix=snapshots_matrix.copy()*0.0

    current_model = KMP.Model()
    model_part = current_model.CreateModelPart("main_model_part")
    create_out_mdpa(model_part, "datasets_rubber_hyperelastic_cantilever_big_range/rubber_hyperelastic_cantilever")

    print_results_to_gid(model_part, snapshots_matrix, reactions_matrix)