import numpy as np

class Base_PrePostProcessor():

    def get_training_data(self, arch_config):
            
            target_aux=None
            val_target_aux=None

            print(self.dataset_path)

            if arch_config["augmented"]:
                S_train_raw=np.load(self.dataset_path+'S_augm_train.npy')
                S_target_train=S_train_raw.copy()
                if arch_config["opt_strategy"]["r_loss_type"]=='norm':
                    print("Can't do RNorm loss with augmented")
                    exit()
                elif arch_config["opt_strategy"]["r_loss_type"]=='diff':
                    target_aux=np.load(self.dataset_path+'R_augm_train.npy')
                else:
                    print('Invalid R loss type at data import')
            else:
                S_train_raw=np.load(self.dataset_path+'S_train.npy')
                # S_train_raw_2=np.load(self.dataset_path+'S_train_extra.npy')
                # S_train_raw=np.concatenate([S_train_raw_1,S_train_raw_2],axis=0)
                S_target_train=S_train_raw.copy()
                if arch_config["opt_strategy"]["r_loss_type"]=='norm':
                    target_aux=np.load(self.dataset_path+'F_train.npy')
                    # target_aux_2=np.load(self.dataset_path+'F_train_extra.npy')
                    # target_aux=np.concatenate([target_aux_1,target_aux_2],axis=0)
                elif arch_config["opt_strategy"]["r_loss_type"]=='diff':
                    target_aux=np.load(self.dataset_path+'R_train.npy')
                    # target_aux_2=np.load(self.dataset_path+'R_train_extra.npy')
                    # target_aux=np.concatenate([target_aux_1,target_aux_2],axis=0)
                else:
                    print('Invalid R loss type at data import')

            input_data, _ = self.preprocess_input_data(S_train_raw)
            target_data=(S_target_train, target_aux)

            if arch_config["augmented"]:
                S_val_raw=np.load(self.dataset_path+'S_augm_val.npy')
                S_target_val=S_val_raw.copy()
                if arch_config["opt_strategy"]["r_loss_type"]=='norm':
                    print("Can't do RNorm loss with augmented")
                    exit()
                elif arch_config["opt_strategy"]["r_loss_type"]=='diff':
                    val_target_aux=np.load(self.dataset_path+'R_augm_val.npy')
                else:
                    print('Invalid R loss type at data import')
            else:
                S_val_raw=np.load(self.dataset_path+'S_val.npy')
                S_target_val=S_val_raw.copy()
                if arch_config["opt_strategy"]["r_loss_type"]=='norm':
                    val_target_aux=np.load(self.dataset_path+'F_val.npy')
                elif arch_config["opt_strategy"]["r_loss_type"]=='diff':
                    val_target_aux=np.load(self.dataset_path+'R_val.npy')
                else:
                    print('Invalid R loss type at data import')

            val_input, _ =self.preprocess_input_data(S_val_raw)
            val_target=(S_target_val, val_target_aux)

            return input_data, target_data, val_input, val_target