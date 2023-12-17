import numpy as np
import tensorflow as tf

from ArchitectureFactories.base_factory import Base_Architecture_Factory

from Optimization_Strategies.s_r_mixed_strategy import S_R_Mixed_Strategy_KerasModel
from Optimization_Strategies.s_only_strategy import S_Only_Strategy_KerasModel
from Optimization_Strategies.r_only_strategy import R_Only_Strategy_KerasModel
from Optimization_Strategies.no_tf_train_strategy import NoTFTrain_Strategy_KerasModel

from PrePostProcessors.Quad_prepost_processors import ScaleGlobal_TF_Quad_PrePostProcessor, Identity_Quad_PrePostProcessor

class Quad_Architecture_Factory(Base_Architecture_Factory):

    def __init__(self, working_path, arch_config):
        super().__init__(working_path, arch_config)

    def arch_config_check(self):
        default_config = {
            "name": 'Quad',
            "q_size": 6,
            "opt_strategy": {},
            "finetune_from": '',
            "augmented": False,
            "prepost_process": ''
        }
 
        # Finding the keys that are missing in actual config compared to default
        diff = set(self.arch_config.keys()).difference(default_config.keys())
        if len(diff)>0:
            print("Architecure config is missing keys or has extra ones compared to default")
            print(diff)
            print("Aborting execution")
            exit()

        for key in default_config.keys():
            if key=="finetune_from":
                if not (type(self.arch_config[key]) is type(default_config[key])) and not (self.arch_config[key] is None):
                    print("Architecure's 'finetune_from' value is not valid")
                    print("Aborting execution")
                    exit()
            else:
                if not type(self.arch_config[key]) is type(default_config[key]):
                    print("Architecure's '"+key+"' value is not valid")
                    print("Aborting execution")
                    exit()

    def generate_model_name_part(self):
        opt_strategy = self.arch_config["opt_strategy"]
        name_part='Quad_'+opt_strategy["name"]+'_'
        if 'tf_' in opt_strategy["name"]:
            name_part+=opt_strategy["r_loss_type"]+'_'
        name_part+=self.arch_config["prepost_process"]+'_'
        if self.arch_config["augmented"]:
            name_part+='Augmented_'
        name_part+='Emb'+str(self.arch_config["q_size"])
        if 'tf_' in opt_strategy["name"]:
            name_part+='_'+'LR'+str(opt_strategy["learning_rate"][0])+str(opt_strategy["learning_rate"][1])
        return self.arch_config["name"], name_part
    
    def keras_model_selector(self, strategy_name):
        if strategy_name=='tf_srmixed':
            print('Using Quad Architecture with S_R_Mixed strategy')
            return S_R_Mixed_Strategy_KerasModel
        if strategy_name=='tf_sonly':
            print('Using Quad Architecture with S_Only strategy')
            return S_Only_Strategy_KerasModel
        if strategy_name=='tf_ronly':
            print('Using Quad Architecture with R_Only strategy')
            return R_Only_Strategy_KerasModel
        if strategy_name=='least_squares':
            print('Using Quad Architecture with LeastSquares strategy')
            return NoTFTrain_Strategy_KerasModel
        else:
            print('No valid ae model was selected')
            return None
        
    def prepost_processor_selector(self, working_path, dataset_path):
        if self.arch_config["prepost_process"] == 'scale_global':
            prepost_processor = ScaleGlobal_TF_Quad_PrePostProcessor(working_path, dataset_path)
        elif self.arch_config["prepost_process"] == 'identity':
            prepost_processor = Identity_Quad_PrePostProcessor(working_path, dataset_path)
        else:
            prepost_processor = None
        return prepost_processor
        
    def configure_prepost_processor(self, prepost_processor, S_flat_orig, crop_mat_tf, crop_mat_scp):
        prepost_processor.configure_processor(S_flat_orig, crop_mat_tf, crop_mat_scp, self.arch_config["q_size"])
        return
    
    def get_custom_LR_scheduler(self):
        if 'tf_' in self.arch_config["opt_strategy"]["name"]:
            return self.get_custom_LR_scheduler_TF()
        else:
            return self.get_custom_LR_scheduler_NoTF()
    
    def define_network(self, prepost_processor, kratos_simulation):

        keras_submodel=self.keras_model_selector(self.arch_config["opt_strategy"]["name"])

        phi_init, h_mat_init = prepost_processor.get_phi_and_H()

        decoded_size = int(phi_init.shape[0])
        encoded_size = int(self.arch_config["q_size"])

        enc_network_in = tf.keras.Input(shape=(decoded_size), dtype=tf.float64)
        dec_network_in = tf.keras.Input(shape=(encoded_size), dtype=tf.float64)
        
        phi = tf.constant(phi_init, dtype=tf.float64, name='Phi')

        if 'tf_' in self.arch_config["opt_strategy"]["name"]:
            if self.arch_config["opt_strategy"]["add_init_noise"]:
                h_mat_init+=np.random.normal(loc=0.0, scale=1e-6, size=h_mat_init.shape)
        
        h_mat = tf.Variable(initial_value=h_mat_init, trainable=True, name='H_matrix', dtype=tf.float64)
        
        encoder_out = tf.transpose(tf.linalg.matmul(phi,enc_network_in,transpose_a=True,transpose_b=True), name='enc_network_out')
        
        out_pr = tf.matmul(tf.expand_dims(dec_network_in,axis=2), tf.expand_dims(dec_network_in,axis=1))
        ones = tf.ones((encoded_size,encoded_size))
        mask = tf.cast(tf.linalg.band_part(ones,0,-1), dtype=tf.bool)
        quad_q = tf.boolean_mask(out_pr, mask, axis=1)

        lin_part = tf.linalg.matmul(phi,dec_network_in,transpose_a=False,transpose_b=True)
        quad_part_pure = tf.linalg.matmul(h_mat,quad_q,transpose_a=False,transpose_b=True)
        quad_part_proj_aux=tf.linalg.matmul(phi,quad_part_pure,transpose_a=True,transpose_b=False)
        quad_part_proj=tf.linalg.matmul(phi,quad_part_proj_aux,transpose_a=False,transpose_b=False)

        decoder_out_aux = tf.math.add(lin_part, quad_part_pure)
        decoder_out = tf.transpose(tf.math.add(decoder_out_aux, tf.math.negative(quad_part_proj), name='dec_network_out'))

        enc_network = tf.keras.Model(enc_network_in, encoder_out, name='enc_network')
        dec_network = tf.keras.Model(dec_network_in, decoder_out, name='dec_network')

        network = keras_submodel(prepost_processor, kratos_simulation, self.arch_config["opt_strategy"], enc_network_in, dec_network(enc_network(enc_network_in)), name='Quadratic_Model')

        network._trainable_weights.append(h_mat)

        network.generate_gradient_sum_functions()
        
        network.compile(optimizer='sgd', run_eagerly=False)
        network.summary()

        return network, enc_network, dec_network
    

    def NMROM_encoder(self, prepost_processor, enc_network):
        def encode_function(s):
            s_norm, _ = prepost_processor.preprocess_input_data(np.expand_dims(s, axis=0))
            q = enc_network(s_norm).numpy()
            return q, None
        return encode_function
    
    def NMROM_decoder(self, prepost_processor, dec_network):
        def decode_function(q, aux_norm_data):
            s_pred_norm = dec_network(q).numpy()
            s_pred = prepost_processor.postprocess_output_data(s_pred_norm, None)
            return s_pred
        return decode_function
    
    def NMROM_decoder_gradient(self, prepost_processor, dec_network):
        def get_decoder_gradient(q):
            @tf.function
            def _get_network_gradient_tf(q):
                with tf.GradientTape(persistent=True) as tape_d:
                    tape_d.watch(q)
                    s_norm=dec_network(q,training=False)
                network_gradient=tape_d.batch_jacobian(s_norm, q, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
                del tape_d
                return network_gradient
            
            network_gradient = _get_network_gradient_tf(q).numpy()[0].T
            decoder_gradient = prepost_processor.postprocess_output_data(network_gradient, None).T
            return decoder_gradient
        return get_decoder_gradient