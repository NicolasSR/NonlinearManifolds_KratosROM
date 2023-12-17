import numpy as np
import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.gid_output_process as GOP

def create_out_mdpa(model_part, file_name):
    model_part.AddNodalSolutionStepVariable(KMP.DISPLACEMENT)
    model_part.AddNodalSolutionStepVariable(KMP.REACTION)

    import_flags = KMP.ModelPartIO.READ

    KMP.ModelPartIO(file_name, import_flags).ReadModelPart(model_part)

def print_results_to_gid(model_part, output_filename, snapshot_matrix, residuals_matrix):

    gid_output = GOP.GiDOutputProcess(
        model_part,
        output_filename,
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

        model_part.ProcessInfo[KMP.STEP] = ts+1
        model_part.ProcessInfo[KMP.TIME] = ts+1
        gid_output.ExecuteBeforeSolutionLoop()
        gid_output.ExecuteInitializeSolutionStep()

        snapshot = snapshot_matrix[ts]
        residuals = residuals_matrix[ts]
        
        i=0
        c=2
        for node in model_part.Nodes:
            node.SetSolutionStepValue(KMP.DISPLACEMENT_X,0,snapshot[i*c+0])
            node.SetSolutionStepValue(KMP.DISPLACEMENT_Y,0,snapshot[i*c+1])
            node.SetSolutionStepValue(KMP.REACTION_X,0,residuals[i*c+0])
            node.SetSolutionStepValue(KMP.REACTION_Y,0,residuals[i*c+1])
            i+=1

        gid_output.PrintOutput()
        gid_output.ExecuteFinalizeSolutionStep()

    gid_output.ExecuteFinalize()

def output_GID_from_matrix(mdpa_filename, output_filename, snapshots_matrix, reactions_matrix):
    current_model = KMP.Model()
    model_part = current_model.CreateModelPart("main_model_part")
    create_out_mdpa(model_part, mdpa_filename)

    print_results_to_gid(model_part, output_filename, snapshots_matrix, reactions_matrix)