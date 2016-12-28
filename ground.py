from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D
from keras.optimizers import SGD,Adam,Adadelta,Nadam
from keras.utils import np_utils
import keras
import math
import keras.backend as K
import sgdr
import numpy as np
from keras.models import load_model
import cv2

# from keras.applications.vgg16 import VGG16

# VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None,None,3))

model = load_model('small_cifar.h5')
model.summary()

flower = cv2.imread('../opencv_playground/flower.jpg').astype('float32')/255.
flowers = flower.reshape((1,)+flower.shape)
target = model.layers[5]
newmodel = Model(input=model.input,output=target.output)
nm=newmodel
