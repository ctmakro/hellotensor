import dicom
import numpy as np
import scipy
import scipy.ndimage
import cv2
import os
import time

visualization = False

if visualization:
    import vis

import threading
l = threading.Lock()

home = '/Users/chia'
DICOM_DIR = home + '/dsb_sample_images/'

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest',order=2)

    return image, new_spacing

def one_subject(subject_id):
    # Read in the directory of a single subject and return a 4d tensor
    directory = os.path.join(DICOM_DIR, subject_id)
    files = [os.path.join(directory, fname)
             for fname in os.listdir(directory) if fname.endswith('.dcm')]

    if len(files)<2 :
        print('not enough file in this directory') # usually not happening
        return None

    slices = [dicom.read_file(fname) for fname in files]
    # each one represents a .dcm file

    ####
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    ####

    # dcm_slices = sorted(dcm_slices, key=lambda x:x.InstanceNumber)

    length = len(slices)

    t3d = np.stack([s.pixel_array for s in slices]).astype('int16')
    t3d[t3d == -2000] = 0
    t3d = t3d.astype('float32')

    # Convert to Hounsfield units (HU)
    # per https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    intercepts = np.stack([s.RescaleIntercept for s in slices]).reshape((length,1,1))
    slopes = np.stack([s.RescaleSlope for s in slices]).reshape((length,1,1))

    t3d = t3d * slopes + intercepts

    # resampling
    t3d, spacing = resample(t3d, slices, [1.8,1.8,1.8])

    # rescaling
    t3d = (np.tanh(t3d/250.)+1.) * 127.5
    # -250,250 -> -1,1 -> 0,255

    t3d = np.clip(t3d,a_min=0.,a_max=255.)

    t3d = t3d.astype('uint8')
    # 3d tensor
    return t3d.reshape((t3d.shape+(1,))) # 4d tensor, last dimension is channel

def all_subject():
    # read all subjects into list of 4d tensor
    directory = DICOM_DIR
    subdirs = os.listdir(directory)
    subdirs = filter(lambda subdir:len(subdir)>10,subdirs)
    # long folder names are hashes

    def read_parallel(subdir):
        l.acquire()
        print('reading from',subdir,'...')
        l.release()
        tick = time.time()
        t4d = one_subject(os.path.join(directory,subdir))
        if t4d is not None: # opposite case unlikely
            # t4d = normalize(t4d) # into (0,1)
            tick = time.time()-tick
            l.acquire()
            print('readed:',t4d.shape,'in {:6.2f} seconds'.format(tick))
            l.release()

            return (t4d,subdir)

    tick2 =time.time()
    if True:
        import thready
        results = thready.amap(read_parallel,subdirs) # parrallel power
    else:
        results = [read_parallel(subdir) for subdir in subdirs]

    tick2 = time.time()-tick2
    print('took {:6.2f} seconds reading {} instances'
    .format(tick2,len(results)))

    tensors=[]
    hashes=[]
    for k in results:
        tensors.append(results[k][0])
        hashes.append(results[k][1])

    return tensors, hashes # 4d tensor list and hash list

# def normalize(t4d):
#     raw = t4d
#     raw_center = raw[:,128:384,128:384,:]
#     upper = raw_center.max()
#     lower = raw_center.min()
#     remap = (raw - lower)/(upper-lower) # into (0,1)
#     remap = np.clip(remap,a_max=1.,a_min=0.)
#     return remap

def disp(img):
    vis.autoscaler_show(img)

tensors, hashes = all_subject()

def loopall():
    for i in range(len(tensors)):
        for j in range(tensors[i].shape[0]):
            disp(tensors[i][j])
        cv2.destroyAllWindows()

def saveall():
    f = open('dicom_tensors.npz','w')
    np.savez(f,tensors=tensors,hashes=hashes)
    f.close()
    print('saved.')

if visualization:
    loopall()
