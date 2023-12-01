import numpy as np
import os
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.initializers import HeNormal

from ArchitectureFactories.base_factory import Base_Architecture_Factory

from Optimization_Strategies.s_r_mixed_strategy import S_R_Mixed_Strategy_KerasModel
from Optimization_Strategies.s_only_strategy import S_Only_Strategy_KerasModel
from Optimization_Strategies.r_only_strategy import R_Only_Strategy_KerasModel
from Optimization_Strategies.w_only_strategy import W_Only_Strategy_KerasModel

from PrePostProcessors.PODANN_prepost_processors import SVD_White_NoStand_PODANN_PrePostProcessor

class BasicKerasEvaluator():

    def __init__(self, working_path, model_path, best):
        self.working_path=working_path
        self.model_path=working_path+model_path
        self.results_path=self.model_path+'reconstruction_evaluation_results/'
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx_'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr_'
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
            self.best_name_part=''
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        with open(self.model_path+"train_config.npy", "rb") as train_config_file:
            self.train_config = np.load(train_config_file,allow_pickle='TRUE').item()
        print(self.train_config)
        self.dataset_path=working_path+self.train_config['dataset_path']

        self.arch_config=self.train_config['architecture']

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def define_network(self):

        use_bias = self.arch_config["use_bias"]

        input_size = self.arch_config["q_inf_size"]
        output_size = self.arch_config["q_sup_size"]-self.arch_config["q_inf_size"]

        decod_input = tf.keras.Input(shape=(input_size,), dtype=tf.float64)

        decoder_out = decod_input
        for layer_size in self.arch_config["hidden_layers"]:
            decoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias, dtype=tf.float64)(decoder_out)
        decoder_out = tf.keras.layers.Dense(output_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias, dtype=tf.float64)(decoder_out)

        network = tf.keras.Model(decod_input, decoder_out, name='q_sup_estimator')

        network.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=False, loss='mse')

        network.summary()

        return network, None, network
    
    
    def execute_evaluation(self):

        print('======= Instantiating TF Model =======')
        self.network, _, __ = self.define_network()
        
        print('======= Loading TF Model weights =======')
        self.network.load_weights(self.model_weights_path+self.model_weights_filename)

        # S_test=np.load(self.dataset_path+'S_test.npy')
        Q_in=np.load('Q_inf_test_matrix.npy')
        Q_out=np.load('Q_sup_test_matrix.npy')

        # S_test, R_test, F_test = self.prepare_evaluation_data()
        # print('Shape S_test: ', S_test.shape)
        # print('Shape R_test: ', R_test.shape)
        # print('Shape F_test: ', F_test.shape)

        self.network.evaluate(x=Q_in, y=Q_out, batch_size=1)

        Q_out_pred=self.network(Q_in).numpy()
        Q_error = np.mean(np.abs(Q_out_pred-Q_out),axis=0)
        # Q_rel_error = np.mean(np.abs(Q_out_pred-Q_out),axis=0)

        plt.plot(Q_error)
        plt.xlabel("mode")
        plt.ylabel("Abs error for the mode, averaged over all samples")
        plt.title("Trained via residuals")

        plt.show()

if __name__=="__main__":

    # model_path = 'PODANN/PODANN_tf_sonly_diff_svd_white_nostand_Lay[40, 40]_Emb6.20_LRtri20.001'
    model_path = 'PODANN/PODANN_tf_ronly_diff_noLog_svd_white_nostand_Lay[40, 40]_Emb6.20_LRtri20.001_lrscale10'

    evaluator=BasicKerasEvaluator('', 'saved_models/'+model_path+'/', 'r')
    evaluator.execute_evaluation()