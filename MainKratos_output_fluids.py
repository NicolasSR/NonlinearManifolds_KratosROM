import sys
import time
import importlib

import KratosMultiphysics
import KratosMultiphysics.FluidDynamicsApplication as FDA

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
            # self.in_velocities_matrix = []
            
            self.computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()

        def FinalizeSolutionStep(self):
            
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

            snapshot = []
            for node in self.computing_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y))
            self.snapshots_matrix.append(snapshot)
                    
        def Finalize(self):
            super().Finalize()

            np.save("FOM_fluids_dataset.npy",           self.snapshots_matrix)
            # np.save("FOM_BoundaryCond.npy", self.in_velocities_matrix)

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    with open("datasets_fluid_past_cylinder/ProjectParameters_FOM.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
