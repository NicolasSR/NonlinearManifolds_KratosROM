import numpy as np
import json


if __name__ == "__main__":

    model_path = 'PODANN/Cont_CorrDecoder_SLoss_Dense_extended_Augmented_SVDWhiteNonStand_emb6_qsup20_lay40_LRsteps_1000ep'

    with open('saved_models/'+model_path+'/train_config.json', "r") as config_file:
        config=json.load(config_file)
    np.save('saved_models/'+model_path+'/train_config.npy', config)