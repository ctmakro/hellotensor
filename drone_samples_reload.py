import numpy as np

def load_dataset(filename):
    z = np.load(filename+'.npy')
    timg,tgt = z['timg'],z['tgt']
    print(timg.shape,tgt.shape)
    return timg,tgt
