import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


if __name__=="__main__":

    dataset_path='datasets_rubber_hyperelastic_cantilever_big_range'

    S_train = np.load(dataset_path+'/S_train.npy')
    F_train = np.load(dataset_path+'/F_train.npy')

    print(S_train.shape)
    print(F_train.shape)
    
    plt.scatter(F_train[:,0,0],F_train[:,6,1], s=4)

    cropped_zones = np.array([[[1500,3000],[2000,3000]],
        [[-1700,-200],[350,1350]],
        [[-1000,1000],[-2500,-700]]])
    
    ax = plt.gca()
    for rect in cropped_zones:
        ax.add_patch(Rectangle((rect[0,0], rect[1,0]), rect[0,1]-rect[0,0], rect[1,1]-rect[1,0],
             edgecolor = 'red', fill=False,  lw=2))
    
    gap_sample_ids=np.array([])
    for rect in cropped_zones:
        ids=np.squeeze(np.argwhere((F_train[:,0,0]>rect[0,0]) & (F_train[:,0,0]<rect[0,1]) & (F_train[:,6,1]>rect[1,0]) & (F_train[:,6,1]<rect[1,1])),axis=1)
        gap_sample_ids=np.concatenate([gap_sample_ids,ids], axis=0).astype(int)
    
    print(gap_sample_ids.shape)
    plt.scatter(F_train[gap_sample_ids,0,0], F_train[gap_sample_ids,6,1], s=4)

    S_train_gappy = np.delete(S_train, gap_sample_ids, axis=0)
    print(S_train_gappy.shape)


    
        
        
    plt.axis('equal')
    plt.show()