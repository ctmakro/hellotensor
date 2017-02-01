from __future__ import print_function

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import math
import random

# import sgdr
import numpy as np

import cv2

def loaddata():
    npz = np.load('dicom_tensors.npz')
    return npz

npz = loaddata()

tensors,hashes = npz['tensors'],npz['hashes']

def disp(img):
    import vis
    vis.autoscaler_show(img)

def loopall():
    for i in range(len(tensors)):
        for j in range(tensors[i].shape[0]):
            disp(tensors[i][j])
        cv2.destroyAllWindows()
