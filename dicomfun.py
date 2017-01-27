import dicom
import numpy as np
import cv2
import os
import time
import vis

home = '/Users/chia'
DICOM_DIR = home + '/dsb_sample_images/'

def read_subject(subject_id):
    # Read in the directory of a single subject and return a 4d tensor
    directory = os.path.join(DICOM_DIR, subject_id)
    files = [os.path.join(directory, fname)
             for fname in os.listdir(directory) if fname.endswith('.dcm')]

    if len(files)<2 :
        print('not enough file in this directory')
        return None

    dcm_slices = [dicom.read_file(fname) for fname in files]
    # each one represents a .dcm file

    def InstanceNumber(dcm):
        return dcm.InstanceNumber

    dcm_slices = sorted(dcm_slices, key=InstanceNumber)
    t3d = np.array([s.pixel_array for s in dcm_slices]).astype('float32')
    # 3d tensor
    return t3d.reshape((t3d.shape+(1,))) # 4d tensor, last dimension is channel

def all_subject():
    # read all subjects into list of 4d tensor
    directory = DICOM_DIR
    subdirs = os.listdir(directory)
    subdirs = filter(lambda subdir:len(subdir)>10,subdirs)
    # long folder names are hashes

    import threading
    l = threading.Lock()

    def read_parallel(subdir):
        l.acquire()
        print('reading from',subdir,'...')
        l.release()
        tick = time.time()
        t4d = read_subject(os.path.join(directory,subdir))
        if t4d is not None: # opposite case unlikely
            t4d = normalize(t4d) # into (0,1)
            tick = time.time()-tick
            l.acquire()
            print('readed:',t4d.shape,'in {:6.2f} seconds'.format(tick))
            l.release()

            return (t4d,subdir)

    import thready
    tick2 =time.time()
    results = thready.amap(read_parallel,subdirs) # parrallel power

    tick2 = time.time()-tick2
    print('took {:6.2f} seconds reading {} instances'
    .format(tick2,len(results)))

    tensors=[]
    hashes=[]
    for k in results:
        tensors.append(results[k][0])
        hashes.append(results[k][1])

    return tensors, hashes # 4d tensor list and hash list

def normalize(t4d):
    raw = t4d
    raw_center = raw[:,128:384,128:384,:]
    upper = raw_center.max()
    lower = raw_center.min()
    remap = (raw - lower)/(upper-lower) # into (0,1)
    remap = np.clip(remap,a_max=1.,a_min=0.)
    return remap

def disp(img):
    vis.autoscaler_show(img)

tensors, hashes = all_subject()

for i in range(len(tensors)):
    for j in range(tensors[i].shape[0]):
        disp(tensors[i][j])
