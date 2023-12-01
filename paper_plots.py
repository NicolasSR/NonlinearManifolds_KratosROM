import numpy as np
from matplotlib import pyplot as plt



rLoss_galerkin_uErr = [0.00016570047444816174, 7.528659101547164e-08]
sLoss_galerkin_uErr = [1.0, 1.921410464249454e-05]

rLoss_galerkin_rErr = [54.270210957191814, 0.10554194886166827]
sLoss_galerkin_rErr = [1000.0, 21.95635276814271]

sLoss_reconstr_uErr = [5.0965800572956246e-05, 1.1569986325000624e-06]
rLoss_reconstr_uErr = [4.367536710106586e-05, 7.975046860941974e-09]

sLoss_reconstr_rErr = [2472.3903407886028,160.89785468030206]
rLoss_reconstr_rErr = [296.8093214558384, 0.620354890020508]

POD_recons_uErr = [0.0009358944254847696, 5.7e-4, 3.6e-4, 2.3e-4, 1.7e-4, 1.0e-4, 3.4e-5, 2.3e-5, 1.0362969606677837e-06]
POD_recons_rErr = [7992.032653971569, 35.04313196375438]

k_vec=[6,20]
POD_k_vec = [6,7,8,9,10,11,12,13,20]


# plt.plot(k_vec, sLoss_galerkin_uErr, 'o', markersize=4, label="ANN-PROM SLoss")
# plt.plot(k_vec, sLoss_reconstr_uErr, 'o', markersize=4, label="Reconstruction SLoss")
# plt.plot(k_vec, rLoss_reconstr_uErr, 'o', markersize=4, label="Reconstruction RLoss")
# plt.plot(k_vec, rLoss_galerkin_uErr, 'o', markersize=4, label="ANN-PROM RLoss")
# plt.plot(POD_k_vec, POD_recons_uErr, 'o', markersize=4, label="Reconstruction POD")
# plt.semilogy()
# plt.legend()
# plt.show()


plt.plot(k_vec, sLoss_galerkin_rErr, 'o', markersize=4, label="ANN-PROM SLoss")
plt.plot(k_vec, sLoss_reconstr_rErr, 'o', markersize=4, label="Reconstruction SLoss")
plt.plot(k_vec, rLoss_reconstr_rErr, 'o', markersize=4, label="Reconstruction RLoss")
plt.plot(k_vec, rLoss_galerkin_rErr, 'o', markersize=4, label="ANN-PROM RLoss")
plt.plot(k_vec, POD_recons_rErr, 'o', markersize=4, label="Reconstruction POD")
plt.semilogy()
plt.legend()
plt.show()



