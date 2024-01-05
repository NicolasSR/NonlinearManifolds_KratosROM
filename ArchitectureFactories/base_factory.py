import abc
import numpy as np

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

from Schedulers.lr_scheduler import get_lr_schedule_func, LR_Scheduler
from Schedulers.lr_wx_wr_scheduler import LR_WX_WR_Scheduler

class Base_Architecture_Factory(abc.ABC):

    def __init__(self, working_path, arch_config):
        super().__init__()
        self.working_path = working_path
        self.arch_config = arch_config
        self.arch_config_check()

        self.model_path=None

    @abc.abstractmethod
    def arch_config_check(self):
        'Defined in the subclasses'

    @abc.abstractmethod
    def generate_model_name_part(self):
        'Defined in the subclasses'
    
    @abc.abstractmethod
    def keras_model_selector(self,ae_config, keras_default):
        'Defined in the subclasses'
        
    @abc.abstractmethod
    def define_network(self, input_data, ae_config, keras_default=False):
        'Defined in the subclasses'

    def set_actual_model_path(self, model_path):
        self.model_path = model_path

    def get_orig_fom_snapshots(self, dataset_path):
        S_FOM_orig=np.load(self.working_path+dataset_path+'FOM.npy')
        return S_FOM_orig

    # def get_training_data(self, prepost_processor, dataset_path):
        
    #     target_aux=None
    #     val_target_aux=None

    #     print(dataset_path)

    #     if self.arch_config["augmented"]:
    #         S_train_raw=np.load(self.working_path+dataset_path+'S_augm_train.npy')
    #         # S_target_train=np.load(self.working_path+dataset_path+'S_repeat_train.npy')
    #         S_target_train=S_train_raw.copy()
    #         if self.arch_config["opt_strategy"]["r_loss_type"]=='norm':
    #             # target_aux=np.load(self.working_path+dataset_path+'F_repeat_train.npy')
    #             print("Can't do RNorm loss with augmented")
    #             exit()
    #         elif self.arch_config["opt_strategy"]["r_loss_type"]=='diff':
    #             # target_aux=np.load(self.working_path+dataset_path+'R_repeat_train.npy')
    #             target_aux=np.load(self.working_path+dataset_path+'R_augm_train.npy')
    #         else:
    #             print('Invalid R loss type at data import')
    #     else:
    #         S_train_raw=np.load(self.working_path+dataset_path+'S_train.npy')
    #         # S_train_raw_2=np.load(self.working_path+dataset_path+'S_train_extra.npy')
    #         # S_train_raw=np.concatenate([S_train_raw_1,S_train_raw_2],axis=0)
    #         S_target_train=S_train_raw.copy()
    #         if self.arch_config["opt_strategy"]["r_loss_type"]=='norm':
    #             target_aux=np.load(self.working_path+dataset_path+'F_train.npy')
    #             # target_aux_2=np.load(self.working_path+dataset_path+'F_train_extra.npy')
    #             # target_aux=np.concatenate([target_aux_1,target_aux_2],axis=0)
    #         elif self.arch_config["opt_strategy"]["r_loss_type"]=='diff':
    #             target_aux=np.load(self.working_path+dataset_path+'R_train.npy')
    #             # target_aux_2=np.load(self.working_path+dataset_path+'R_train_extra.npy')
    #             # target_aux=np.concatenate([target_aux_1,target_aux_2],axis=0)
    #         else:
    #             print('Invalid R loss type at data import')

    #     input_data = prepost_processor.preprocess_input_data(S_train_raw)
    #     target_data=(S_target_train, target_aux)

    #     if self.arch_config["augmented"]:
    #         S_val_raw=np.load(self.working_path+dataset_path+'S_augm_val.npy')
    #         S_target_val=S_val_raw.copy()
    #         if self.arch_config["opt_strategy"]["r_loss_type"]=='norm':
    #             print("Can't do RNorm loss with augmented")
    #             exit()
    #         elif self.arch_config["opt_strategy"]["r_loss_type"]=='diff':
    #             val_target_aux=np.load(self.working_path+dataset_path+'R_augm_val.npy')
    #         else:
    #             print('Invalid R loss type at data import')
    #     else:
    #         S_val_raw=np.load(self.working_path+dataset_path+'S_val.npy')
    #         S_target_val=S_val_raw.copy()
    #         if self.arch_config["opt_strategy"]["r_loss_type"]=='norm':
    #             val_target_aux=np.load(self.working_path+dataset_path+'F_val.npy')
    #         elif self.arch_config["opt_strategy"]["r_loss_type"]=='diff':
    #             val_target_aux=np.load(self.working_path+dataset_path+'R_val.npy')
    #         else:
    #             print('Invalid R loss type at data import')

    #     val_input=prepost_processor.preprocess_input_data(S_val_raw)
    #     val_target=(S_target_val, val_target_aux)

    #     return input_data, target_data, val_input, val_target

    def get_custom_LR_scheduler_TF(self):
        opt_strategy_config=self.arch_config["opt_strategy"]
        
        if opt_strategy_config["name"]=='tf_srmixed' or opt_strategy_config["name"]=='tf_srmixed_cropped':
            lr_schedule_func = get_lr_schedule_func(opt_strategy_config["learning_rate"])
            wx_schedule_func = get_lr_schedule_func(opt_strategy_config["wx"])
            wr_schedule_func = get_lr_schedule_func(opt_strategy_config["wr"])
            print('Using scheduler for LR, wx and wr')
            return LR_WX_WR_Scheduler(lr_schedule_func, wx_schedule_func, wr_schedule_func)
        else:
            ## Basic scheduler, just for the learning rate itself
            lr_schedule_func = get_lr_schedule_func(opt_strategy_config["learning_rate"])
            print('Using scheduler for LR alone')
            return LR_Scheduler(lr_schedule_func)
        
    def get_custom_LR_scheduler_NoTF(self):
        lr_schedule_func = get_lr_schedule_func(('const',1e-4))
        return LR_Scheduler(lr_schedule_func)

    def train_network(self, model, input_data, target_data, val_input, val_target):
        # Train the model

        # early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss_r', patience=5)
        
        model.update_rescaling_factors(target_data[0], target_data[1])

        # checkpoint_best_x_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"best/weights_x_{epoch:03d}.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_x",mode="min")
        checkpoint_best_x_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"best/weights_x_best.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_x",mode="min")
        # checkpoint_best_r_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"best/weights_r_{epoch:03d}.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_r",mode="min")
        checkpoint_best_r_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"best/weights_r_best.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_r",mode="min")
        checkpoint_last_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"last/weights.h5",save_weights_only=True,save_freq="epoch")
        lr_scheduler_callback = self.get_custom_LR_scheduler()
        csv_logger_callback = tf.keras.callbacks.CSVLogger(self.model_path+"train_log.csv", separator=',', append=False)

        if  not "batch_size" in self.arch_config["opt_strategy"]:
            self.arch_config["opt_strategy"]["batch_size"] = 1
        if  not "epochs" in self.arch_config["opt_strategy"]:
            self.arch_config["opt_strategy"]["epochs"] = 1

        history = model.fit(
            input_data, target_data,
            epochs=self.arch_config["opt_strategy"]["epochs"],
            shuffle=True,
            # shuffle=False,
            batch_size=self.arch_config["opt_strategy"]["batch_size"],
            validation_data=(val_input,val_target),
            validation_batch_size=1,
            callbacks = [
                lr_scheduler_callback,
                checkpoint_best_x_callback,
                checkpoint_best_r_callback,
                checkpoint_last_callback,
                csv_logger_callback
            ]
        )

        return history