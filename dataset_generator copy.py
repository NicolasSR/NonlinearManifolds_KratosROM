import sys
import time
import importlib

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import numpy as np

from scipy.stats import qmc

from matplotlib import pyplot as plt

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()

            self.mu_counter=0
            self.mu=self.get_mu_parameters()
            self.project_parameters["problem_data"]["end_time"].SetInt(len(self.mu))

            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()
            
        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()

            self.snapshots_matrix = []
            self.pointload_matrix = []

            self.main_model_part = self.model.GetModelPart("Structure")
            self.point_load_h_model_part = self.model.GetModelPart("Structure.LineLoad2D_ForceH")
            self.point_load_v_model_part = self.model.GetModelPart("Structure.LineLoad2D_ForceV")

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
            point_load_h = KratosMultiphysics.Vector(np.array([self.mu[self.mu_counter][0],0.0 ,0.0]))
            for condition in self.point_load_h_model_part.Conditions:
                condition.SetValue(SMA.LINE_LOAD, point_load_h)
            point_load_v = KratosMultiphysics.Vector(np.array([0.0, self.mu[self.mu_counter][1] ,0.0]))
            for condition in self.point_load_v_model_part.Conditions:
                condition.SetValue(SMA.LINE_LOAD, point_load_v)

            print(self.mu[self.mu_counter])
            self.mu_counter += 1

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

        def get_mu_parameters(self):
            sampler = qmc.Halton(d=2, seed=1)
            mu = sampler.random(n=5000)
            # mu=qmc.scale(mu, [-600, -600], [0.0,0.0])
            mu=[[-600, -600],[-23,-153],[0.0,0.0],[-600,-20]]
            # mu=[[-263.85526782, -533.7345887]]

            # mu1=np.linspace(0.0,600,10)*(-1)
            # mu2=np.linspace(0.0,600,10)*(-1)
            # mu=np.concatenate((mu1, mu2)).reshape((-1, 2), order='F')

            # print(mu)
            # plt.scatter(mu[:,0],mu[:,1], s=1)
            # plt.show()
            return mu


            

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    dataset_path='datasets_rubber_hyperelastic_cantilever/'

    with open(dataset_path+"ProjectParameters_FOM.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()