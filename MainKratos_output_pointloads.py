import sys
import time
import importlib

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import numpy as np

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
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

            np.save("FOM.npy",           self.snapshots_matrix)
            np.save("FOM_POINTLOADS.npy", self.pointload_matrix)

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    with open("datasets_rubber_hyperelastic_cantilever_big_range/ProjectParameters_recons.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
