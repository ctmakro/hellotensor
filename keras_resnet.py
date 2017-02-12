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

import sgdr
import numpy as np

def bn(x):
    # return x
    return BatchNormalization()(x)
def relu(x):
    return LeakyReLU(alpha=.2)(x)
    return Activation('tanh')(x)
    return Activation('relu')(x)

# i = Deconvolution2D(128,3,3,output_shape=(8,8,128),border_mode='same')(i)

def gen_neck(nip,nop,stride,h,w,bsize): # h and w are input height and width
    def unit(x):
        nBottleneckPlane = int(max(nip,nop)/4)
        nbp = nBottleneckPlane

        if nip==nop and stride==1:
            ident = x

            x = bn(x)
            x = relu(x)
            # i = Deconvolution2D(32,5,5,subsample=(2,2),output_shape=(batch_size,16,16,32),border_mode='same')(i)

            x = Deconvolution2D(nbp,1,1,subsample=(stride,stride),output_shape=(bsize,h*stride,w*stride,nbp),border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Deconvolution2D(nbp,3,3,output_shape=(bsize,h*stride,w*stride,nbp),border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Deconvolution2D(nop,1,1,output_shape=(bsize,h*stride,w*stride,nop),border_mode='same')(x)

            out = merge([ident,x],mode='sum')
        else:
            x = bn(x)
            x = relu(x)
            ident = x

            x = Deconvolution2D(nbp,1,1,subsample=(stride,stride),output_shape=(bsize,h*stride,w*stride,nbp),border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Deconvolution2D(nbp,3,3,output_shape=(bsize,h*stride,w*stride,nbp),border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Deconvolution2D(nop,1,1,output_shape=(bsize,h*stride,w*stride,nop),border_mode='same')(x)

            ident = Deconvolution2D(nop,1,1,subsample=(stride,stride),output_shape=(bsize,h*stride,w*stride,nop),border_mode='same')(ident)
            out = merge([ident,x],mode='sum')
        return out
    return unit

# per freshest resnet paper
def neck(nip,nop,stride,disc=False):
    def unit(x):
        nBottleneckPlane = int(max(nip,nop) / 4)
        nbp = nBottleneckPlane

        if nip==nop and stride==1:
            ident = x

            x = bn(x)
            x = relu(x)
            x = Convolution2D(nbp,1,1,
            subsample=(stride,stride))(x)

            x = bn(x)
            x = relu(x)
            x = Convolution2D(nbp,3,3,border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Convolution2D(nop,1,1)(x)

            out = merge([ident,x],mode='sum')
        else:
            x = bn(x)
            x = relu(x)
            ident = x

            x = Convolution2D(nbp,1,1,
            subsample=(stride,stride))(x)

            x = bn(x)
            x = relu(x)
            x = Convolution2D(nbp,3,3,border_mode='same')(x)

            x = bn(x)
            x = relu(x)
            x = Convolution2D(nop,1,1)(x)

            ident = Convolution2D(nop,1,1,
            subsample=(stride,stride))(ident)

            out = merge([ident,x],mode='sum')

        return out
    return unit

def cake(nip,nop,layers,std,disc=False):
    def unit(x):
        for i in range(layers):
            if i==0:
                x = neck(nip,nop,std,disc)(x)
            else:
                x = neck(nop,nop,1,disc)(x)
        return x
    return unit

def gen_cake(nip,nop,layers,std,inputh,inputw,bsize):
    def unit(x):
        for i in range(layers):
            if i!=layers-1:
                x = gen_neck(nip,nip,1,inputh,inputw,bsize)(x)
            else:
                x = gen_neck(nip,nop,std,inputh,inputw,bsize)(x)
        return x
    return unit
