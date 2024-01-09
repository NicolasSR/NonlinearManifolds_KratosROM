import numpy as np
import matplotlib.pyplot as plt

models_path='saved_models_cantilever_big_range/'
R_file_path = models_path+'PODANN/PODANN_tf_ronly_Cont_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.0001_slower/NMROM_simulation_results_random300/ROM_residuals_converged_corrected.npy'
# R_file_path = models_path+'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb6.60_LRsgdr0.001_slower/NMROM_simulation_results_random300/ROM_residuals_converged_corrected.npy'


F_test_path = 'datasets_rubber_hyperelastic_cantilever_big_range/F_test_small.npy'

R=np.load(R_file_path)
F_full=np.load(F_test_path)

F=[]
for f in F_full:
    F.append([f[0,0],f[6,1]])
F=np.array(F)

R_norm = np.linalg.norm(R, axis=1)

id = 1
# plt.scatter(F[id,0],F[id,1], s=20, c=R_norm[id])
plt.scatter(F[:,0],F[:,1], s=20, c=R_norm)
plt.colorbar()
plt.show()
