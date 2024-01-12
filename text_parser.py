import argparse
import json

import numpy as np
from matplotlib import pyplot as plt

import re


# ::[ROM Simulation]:: : STEP:  6 
# ::[ROM Simulation]:: : TIME:  6.0 
# ResidualBasedNewtonRaphsonStrategy: System Construction Time: 1.826e-06 [s]
# SolvingStrategy:  MESH MOVED 
# GlobalROMResidualBasedBlockBuilderAndSolver: Build time: 0.00102553
# AnnPromGlobalROMBuilderAndSolver: Build and project time: 0.00114748
# AnnPromGlobalROMBuilderAndSolver: Solve reduced system time: 3.077e-06
# AnnPromGlobalROMBuilderAndSolver: Project to fine basis time: 0.000154993
# SolvingStrategy:  MESH MOVED 
# DISPLACEMENT CRITERION:  :: [ Obtained ratio = 0.158775; Expected ratio = 1e-06; Absolute norm = 0.000665635; Expected norm =  1e-08]


if __name__=="__main__":


   file_path = 'output_log_FOM.txt'

   with open(file_path, 'r') as file:
      data = file.read()

   # solve_times_substring = "Solve reduced system time: "
   solve_times_substring = "System solve time: "
   solve_times_matches = re.findall(f"{solve_times_substring}.+", data)

   print(f"Occurrences of '{solve_times_substring}': {len(solve_times_matches)}")

   solve_times=[]
   for match in solve_times_matches:
      try:
         # solve_times.append(float((f"{match}").replace(solve_times_substring,'')))
         solve_times.append(float((f"{match}").replace(solve_times_substring,'').replace(' [s]','')))
      except:
         pass
   solve_times=np.array(solve_times)

   print('Mean solve time: ', np.mean(solve_times))
   print('Median solve time: ', np.median(solve_times))

   plt.plot(solve_times)
   plt.semilogy()
   plt.show()


   project_times_substring = "Project to fine basis time: "
   project_times_matches = re.findall(f"{project_times_substring}.+", data)

   print(f"Occurrences of '{project_times_substring}': {len(project_times_matches)}")

   project_times=[]
   for match in project_times_matches:
      try:
         project_times.append(float((f"{match}").replace(project_times_substring,'')))
      except:
         pass
   project_times=np.array(project_times)

   print('Mean project time: ', np.mean(project_times))
   print('Median project time: ', np.median(project_times))

   plt.plot(project_times)
   plt.semilogy()
   plt.show()
