import sys
import time
import importlib

import numpy as np

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA


def CreateAnalysisStageWithFlushInstance(cls, snapshot_mat_filename, pointload_mat_filename, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()
            
        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()

            self.snapshots_matrix = []
            self.pointload_matrix = []

            self.main_model_part = self.model.GetModelPart("Structure")

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now
                    
            pointload = []
            for condition in self.main_model_part.Conditions:
                pointload.append(condition.GetValue(SMA.LINE_LOAD))
            self.pointload_matrix.append(pointload)

            snapshot = []
            for node in self.main_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)
                    
        def Finalize(self):
            super().Finalize()

            np.save(snapshot_mat_filename, self.snapshots_matrix)
            np.save(pointload_mat_filename, self.pointload_matrix)

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    dataset_path = "datasets_two_forces_dense_extended/"

    snapshot_mat_filename=dataset_path+"FOM/FOM_for_recons.npy"
    pointload_mat_filename=dataset_path+"FOM/POINTLOADS_for_recons.npy"

    with open(dataset_path+"ProjectParameters_recons_repeatedStep.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, snapshot_mat_filename, pointload_mat_filename, global_model, parameters)
    simulation.Run()
