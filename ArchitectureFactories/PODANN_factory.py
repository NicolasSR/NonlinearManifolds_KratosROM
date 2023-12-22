import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

from ArchitectureFactories.base_factory import Base_Architecture_Factory

from Optimization_Strategies.s_r_mixed_strategy_batches import S_R_Mixed_Strategy_KerasModel
from Optimization_Strategies.s_only_strategy_batches import S_Only_Strategy_KerasModel
from Optimization_Strategies.r_only_strategy_batches import R_Only_Strategy_KerasModel
from Optimization_Strategies.w_only_strategy import W_Only_Strategy_KerasModel
from Optimization_Strategies.s_only_strategy_batches_crop import S_Only_Strategy_Cropped_KerasModel
from Optimization_Strategies.r_only_strategy_batches_crop import R_Only_Strategy_Cropped_KerasModel
from Optimization_Strategies.s_r_mixed_strategy_batches_crop import S_R_Mixed_Strategy_Cropped_KerasModel
from Optimization_Strategies.s_farhat_strategy_batches import S_Farhat_Strategy_KerasModel

from PrePostProcessors.PODANN_prepost_processors import SVD_White_NoStand_PODANN_PrePostProcessor, SVD_Rerange_PODANN_PrePostProcessor, SVD_PODANN_PrePostProcessor
from PrePostProcessors.PODANN_prepost_processors import SVD_White_NoStand_Cropping_PODANN_PrePostProcessor

class PODANN_Architecture_Factory(Base_Architecture_Factory):

    def __init__(self, working_path, arch_config):
        super().__init__(working_path, arch_config)

    def arch_config_check(self):
        default_config = {
            "name": 'PODANN',
            "q_inf_size": 0,
            "q_sup_size": 0,
            "hidden_layers": [],
            "prepost_process": '',
            "opt_strategy": {},
            "finetune_from": '',
            # "finetune_from": None,
            "augmented": False,
            "use_bias": False,
            "use_dropout": 0.1
        }
 
        # Finding the keys that are missing in actual config compared to default
        diff = set(self.arch_config.keys()).difference(default_config.keys())
        if len(diff)>0:
            print("Architecure config is missing keys or has extra ones compared to default")
            print(diff)
            print("Aborting execution")
            exit()

        for key in default_config.keys():
            if key=="finetune_from"  or key=="use_dropout":
                if not (type(self.arch_config[key]) is type(default_config[key])) and not (self.arch_config[key] is None):
                    print("Architecure's ", key, " value is not valid")
                    print("Aborting execution")
                    exit()
            else:
                if not type(self.arch_config[key]) is type(default_config[key]):
                    print("Architecure's '"+key+"' value is not valid")
                    print("Aborting execution")
                    exit()

    def generate_model_name_part(self):
        opt_strategy_config=self.arch_config["opt_strategy"]
        name_part='PODANN_'+opt_strategy_config["name"]+'_'
        if type(self.arch_config["finetune_from"]) is type(''):
            name_part+='Cont'+'_'
        if "r_loss_type" in opt_strategy_config.keys():
            name_part+=opt_strategy_config["r_loss_type"]+'_'
        name_part+=self.arch_config["prepost_process"]+'_'
        if self.arch_config["augmented"]:
            name_part+='Augmented_'
        name_part+='Lay'+str(self.arch_config["hidden_layers"])+'_'
        if not self.arch_config["use_dropout"] is None:
            name_part+='Drop'+str(self.arch_config["use_dropout"])+'_'
        name_part+='Emb'+str(self.arch_config["q_inf_size"])+'.'+str(self.arch_config["q_sup_size"])+'_'
        name_part+='LR'+str(opt_strategy_config["learning_rate"][0])+str(opt_strategy_config["learning_rate"][1])
        return self.arch_config["name"], name_part
    
    def keras_model_selector(self, strategy_name):

        if strategy_name=='tf_srmixed':
            print('Using PODANN Architecture with S_R_Mixed strategy')
            return S_R_Mixed_Strategy_KerasModel
        elif strategy_name=='tf_sonly':
            print('Using PODANN Architecture with S_Only strategy')
            return S_Only_Strategy_KerasModel
        elif strategy_name=='tf_ronly':
            print('Using PODANN Architecture with R_Only strategy')
            return R_Only_Strategy_KerasModel
        elif strategy_name=='tf_sonly_cropped':
            print('Using PODANN Architecture with S_Only_Croped strategy')
            return S_Only_Strategy_Cropped_KerasModel
        elif strategy_name=='tf_ronly_cropped':
            print('Using PODANN Architecture with R_Only_Croped strategy')
            return R_Only_Strategy_Cropped_KerasModel
        elif strategy_name=='tf_srmixed_cropped':
            print('Using PODANN Architecture with S_R_Mixed_Croped strategy')
            return S_R_Mixed_Strategy_Cropped_KerasModel
        elif strategy_name=='tf_wonly':
            print('Using PODANN Architecture with W_Only strategy')
            return W_Only_Strategy_KerasModel
        elif strategy_name=='tf_sfarhat':
            print('Using PODANN Architecture with S_Farhat strategy')
            return S_Farhat_Strategy_KerasModel
        else:
            print('No valid ae model was selected')
            return None
        
    def prepost_processor_selector(self, working_path, dataset_path):
        if self.arch_config["prepost_process"] == 'svd_white_nostand':
            prepost_processor = SVD_White_NoStand_PODANN_PrePostProcessor(working_path, dataset_path)
        elif self.arch_config["prepost_process"] == 'svd_white_nostand_crop':
            prepost_processor = SVD_White_NoStand_Cropping_PODANN_PrePostProcessor(working_path, dataset_path)
        elif self.arch_config["prepost_process"] == 'svd_rerange':
            prepost_processor = SVD_Rerange_PODANN_PrePostProcessor(working_path, dataset_path)
        elif self.arch_config["prepost_process"] == 'svd':
            prepost_processor = SVD_PODANN_PrePostProcessor(working_path, dataset_path)
        else:
            print('Normalization strategy is not valid')
            prepost_processor = None
        return prepost_processor
        
    def configure_prepost_processor(self, prepost_processor, S_flat_orig, crop_mat_tf, crop_mat_scp):
        prepost_processor.configure_processor(S_flat_orig, self.arch_config["q_inf_size"], self.arch_config["q_sup_size"], crop_mat_tf, crop_mat_scp)
        self.snapshot_size=S_flat_orig.shape[1]
        return
    
    def get_custom_LR_scheduler(self):
        return self.get_custom_LR_scheduler_TF()

    def define_network(self, prepost_processor, kratos_simulation):

        keras_submodel=self.keras_model_selector(self.arch_config["opt_strategy"]["name"])

        use_bias = self.arch_config["use_bias"]
        dropout_rate = self.arch_config["use_dropout"]

        input_size = self.arch_config["q_inf_size"]
        output_size = self.arch_config["q_sup_size"]-self.arch_config["q_inf_size"]

        decod_input = tf.keras.Input(shape=(input_size,), dtype=tf.float64)

        decoder_out = decod_input
        for layer_size in self.arch_config["hidden_layers"]:
            decoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias, dtype=tf.float64)(decoder_out)
            if not dropout_rate is None:
                decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(output_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias, dtype=tf.float64)(decoder_out)

        network = keras_submodel(prepost_processor, kratos_simulation, self.arch_config["opt_strategy"], decod_input, decoder_out, name='q_sup_estimator')
        
        network.generate_gradient_sum_functions()

        network.compile(optimizer=tf.keras.optimizers.experimental.AdamW(epsilon=1e-17), run_eagerly=network.run_eagerly)
        # network.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=network.run_eagerly)
        # network.compile(optimizer=tf.keras.optimizers.SGD(), run_eagerly=network.run_eagerly)

        network.summary()

        return network, None, network
    

    def NMROM_encoder(self, prepost_processor, enc_network):
        def encode_function(s):
            q_inf, aux_norm_data = prepost_processor.preprocess_input_data(np.expand_dims(s, axis=0))
            return q_inf, aux_norm_data
        return encode_function
    
    def NMROM_decoder(self, prepost_processor, dec_network):
        def decode_function(q_inf, aux_norm_data):
            q_sup_pred=dec_network(q_inf).numpy()
            s_pred = prepost_processor.postprocess_output_data(q_sup_pred, (q_inf, aux_norm_data))
            return s_pred
        return decode_function
    
    def NMROM_decoder_gradient(self, prepost_processor, dec_network):
        identity_tensor = np.eye(self.arch_config["q_inf_size"])
        zero_tensor = np.zeros((self.snapshot_size))
        def get_decoder_gradient(q_inf):
            @tf.function
            def _get_network_gradient_tf(q_inf):
                with tf.GradientTape(persistent=True) as tape_d:
                    tape_d.watch(q_inf)
                    q_sup=dec_network(q_inf,training=False)
                network_gradient=tape_d.batch_jacobian(q_sup, q_inf, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
                del tape_d
                return network_gradient
            
            network_gradient = _get_network_gradient_tf(q_inf).numpy()[0].T
            decoder_gradient = prepost_processor.postprocess_output_data(network_gradient, (identity_tensor,zero_tensor)).T
            return decoder_gradient
        return get_decoder_gradient
    
    # def NMROM_linear_decoder_gradient(self, prepost_processor, dec_network):
    #     identity_tensor = np.eye(self.arch_config["q_inf_size"])
    #     zero_tensor = np.zeros((self.arch_config["q_sup_size"]-self.arch_config["q_inf_size"],self.arch_config["q_inf_size"])).T
    #     def get_decoder_gradient(q_inf):
    #         decoder_gradient = prepost_processor.postprocess_output_data(zero_tensor, identity_tensor).T
    #         return decoder_gradient
    #     return get_decoder_gradient