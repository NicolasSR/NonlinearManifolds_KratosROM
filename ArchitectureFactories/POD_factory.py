import numpy as np
import tensorflow as tf

from ArchitectureFactories.base_factory import Base_Architecture_Factory

from Optimization_Strategies.no_tf_train_strategy import NoTFTrain_Strategy_KerasModel

from PrePostProcessors.POD_prepost_processors import Identity_POD_PrePostProcessor

class POD_Architecture_Factory(Base_Architecture_Factory):

    def __init__(self, working_path, arch_config):
        super().__init__(working_path, arch_config)

    def arch_config_check(self):

        default_config = {
            "name": 'POD',
            "q_size": 6,
            "augmented": False,
            "opt_strategy": {},
            "finetune_from": ''
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
        name_part='POD_'
        if self.arch_config["augmented"]:
            name_part+='Augmented_'
        name_part+='Emb'+str(self.arch_config["q_size"])
        return self.arch_config["name"], name_part
    
    def keras_model_selector(self):
        print('Using POD Architecture with LeastSquares strategy')
        return NoTFTrain_Strategy_KerasModel
        
    def prepost_processor_selector(self, working_path, dataset_path):
        return Identity_POD_PrePostProcessor(working_path, dataset_path)
        
    def configure_prepost_processor(self, prepost_processor, S_flat_orig, crop_mat_tf, crop_mat_scp):
        prepost_processor.configure_processor(S_flat_orig, self.arch_config["q_size"])
        return
    
    def get_custom_LR_scheduler(self):
        return self.get_custom_LR_scheduler_NoTF()
    
    def define_network(self, prepost_processor, kratos_simulation):

        keras_submodel=self.keras_model_selector()

        phi_init = prepost_processor.get_phi()

        decoded_size = int(phi_init.shape[0])
        encoded_size = int(phi_init.shape[1])

        enc_network_in = tf.keras.Input(shape=(decoded_size), dtype=tf.float64)
        dec_network_in = tf.keras.Input(shape=(encoded_size), dtype=tf.float64)
        
        phi = tf.constant(phi_init, dtype=tf.float64, name='Phi')
        
        encoder_out = tf.transpose(tf.linalg.matmul(phi,enc_network_in,transpose_a=True,transpose_b=True), name='enc_network_out')

        decoder_out = tf.transpose(tf.linalg.matmul(phi,dec_network_in,transpose_a=False,transpose_b=True), name='dec_network_out')

        enc_network = tf.keras.Model(enc_network_in, encoder_out, name='enc_network')
        dec_network = tf.keras.Model(dec_network_in, decoder_out, name='dec_network')

        network = keras_submodel(prepost_processor, kratos_simulation, self.arch_config["opt_strategy"], enc_network_in, dec_network(enc_network(enc_network_in)), name='POD_Model_network')
        
        network.compile(optimizer='sgd', run_eagerly=False)
        network.summary()

        return network, enc_network, dec_network

    def NMROM_encoder(self, prepost_processor, enc_network):
        def encode_function(s):
            s_norm = prepost_processor.preprocess_input_data(np.expand_dims(s, axis=0))
            q = enc_network(s_norm).numpy()
            return q
        return encode_function
    
    def NMROM_decoder(self, prepost_processor, dec_network):
        def decode_function(q):
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
            
            network_gradient = _get_network_gradient_tf(q).numpy()[0]
            decoder_gradient = prepost_processor.postprocess_output_data(network_gradient, None)
            return decoder_gradient
        return get_decoder_gradient
