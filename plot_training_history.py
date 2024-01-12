import argparse
import json

import numpy as np
from matplotlib import pyplot as plt



class Model_Info:
    def __init__(self, working_path, model_path, label):

      self.label=label
      self.working_path = working_path
      self.model_path = working_path+'saved_models_cantilever_big_range/'+model_path

      with open(self.model_path+'history.json', "r") as history_file:
         self.history = json.load(history_file)


def plot_grad_means(model_info_list):

   labels_list=[]
   grads_matrix=[]
   for model_info in model_info_list:
      # plt.plot(model_info.history["loss_x"], label=model_info.label+' train')
      grads_matrix.append(model_info.history["grads_mean"])
      labels_list.append(model_info.label)
   grads_matrix=np.array(grads_matrix).T

   plt.boxplot(grads_matrix)

   ax=plt.gca()
   ax.set_xticklabels(labels_list)
   
   plt.semilogy()
   plt.xlabel("Model (Loss type and q_inf size)")
   plt.ylabel("Mean gradient value per batch over training routine")
   plt.show()

def plot_x_loss(model_info_list):

   for model_info in model_info_list:
      # plt.plot(model_info.history["loss_x"], label=model_info.label+' train')
      plt.plot(model_info.history["val_loss_x"], label=model_info.label+' val')
   
   plt.semilogy()
   plt.legend()
   plt.show()

def plot_r_loss(model_info_list):

   for model_info in model_info_list:
      # plt.plot(model_info.history["loss_r"], label=model_info.label+' train')
      plt.plot(model_info.history["val_loss_r"], label=model_info.label+' val')
   
   plt.semilogy()
   plt.legend()
   plt.show()

if __name__=="__main__":

   result_cases = [{
   #     "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001/',
   #     "label": 'typical'
   #  },{
   #     "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsgdr0.001_slower/',
   #     "label": 'slow'
   #  },{
   #     "model_path": 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[200, 200]_Emb20.60_LRsteps0.001/',
   #     "label": 'steps'
   #  },{
       "model_path": 'PODANN/gradients_collection_ronly_6/',
      #  "model_path": 'PODANN/gradients_collection_ronly_6_norescale/',
       "label": 'R 6'
    },{
       "model_path": 'PODANN/gradients_collection_ronly_20/',
      #  "model_path": 'PODANN/gradients_collection_ronly_20_norescale/',
       "label": 'R 20'
    },{
       "model_path": 'PODANN/gradients_collection_sonly_6/',
      #  "model_path": 'PODANN/gradients_collection_sonly_6_norescale/',
       "label": 'S 6'
    },{
       "model_path": 'PODANN/gradients_collection_sonly_20/',
      #  "model_path": 'PODANN/gradients_collection_sonly_20_norescale/',
       "label": 'S 20'
    }
    ]
   
   parser = argparse.ArgumentParser()
   parser.add_argument('working_path', type=str, help='Root directory to work from.')
   args = parser.parse_args()
   working_path = args.working_path+'/'

   model_info_list=[]
   for case in result_cases:
       model_info_list.append(Model_Info(working_path, case["model_path"], case["label"]))

   plot_grad_means(model_info_list)
   plot_x_loss(model_info_list)
   plot_r_loss(model_info_list)

   print('FINISHED EVALUATING')