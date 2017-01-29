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

import cv2

# keras boilerplate ended

from waves import song,loadfile,show_waterfall,play,fs

eps = 1e-11

def relu(i):
    return Activation('relu')(i)

def wnet(discriminate=False):
    print('building wnet...')
    inp = Input(shape=(1024,2)) # stereo

    i = inp

    def ac(i,featin,featout,kw,ar):
        feat = featout
        ident = i
        qf = max(featin,featout)/4

        i = relu(i)
        i = Convolution1D(qf, 1, border_mode='same')(i)

        i = relu(i)
        i = AtrousConvolution1D(qf, kw, atrous_rate=ar, border_mode='same')(i)

        i = relu(i)
        i = Convolution1D(feat, 1, border_mode='same')(i)

        if featin==featout:
            ident = ident
        else:
            ident = Convolution1D(feat, 1, border_mode='same')(ident)

        i = merge([ident,i],mode='sum')

        return i

    i = Convolution1D(16,3,border_mode='same')(i)
    i = ac(i,16,32,3,1)

    for k in range(2):
        for j in range(7): # 0..6
            i = ac(i,32,32,3,2**(j+1)) # 2**(1..7)

        i = ac(i,32,32,3,256)

        if discriminate==False:
            for j in reversed(range(7)): # 6..0
                i = ac(i,32,32,3,2**(j+1)) # 2**(7..1)

    if discriminate==False:
        i = ac(i,32,16,3,1)
        i = Convolution1D(2, 3, border_mode='same')(i)
        i = Activation('tanh')(i)
    else:
        i = relu(i)
        i = Convolution1D(1, 1, border_mode='valid')(i)
        i = relu(i)
        # i = AveragePooling1D(1024,border_mode='valid')(i)
        i = Convolution1D(1, 1024, border_mode='valid')(i)
        i = Reshape((1,))(i)
        i = Activation('sigmoid')(i)

    out = i
    m = Model(input=inp,output=out)
    mf = Model(input=inp,output=out)
    mf.trainable = False
    return m,mf

print('G...')
gm,gmf = wnet(discriminate=False)
gm.summary()
print('D...')
dm,dmf = wnet(discriminate=True)
dm.summary()

# w.compile(loss='mse',optimizer=Adam(lr=1e-3))

def trainer(gm,gmf,dm,dmf):
    before = Input(shape=(1024,2))
    after = Input(shape=(1024,2))

    gened = gmf(before)

    rs = dm(after)
    gs = dm(gened)

    def ccel(x):
        gs=x[0]
        rs=x[1]
        loss = - (K.log(1-gs+eps) + 0.1 * K.log(1-rs+eps) + 0.9 * K.log(rs+eps)) #sside lbl smoothing
        return loss

    def calc_output_shape(input_shapes):
        return input_shapes[0]

    dloss = merge([gs,rs],mode=ccel,output_shape=calc_output_shape,name='dloss')

    dm_trainer = Model(input=[before,after],output=dloss)

    def thru(y_true,y_pred):
        return y_pred

    gened = gm(before)
    gs = dmf(gened)
    gloss = Lambda(lambda x:- K.log(x+eps),name='gloss')(gs)

    gm_trainer = Model(input=before,output=gloss)

    lr,b1 = 1e-3,.2

    dm_trainer.compile(loss=thru,optimizer=Adam(lr=lr,beta_1=b1))
    gm_trainer.compile(loss=thru,optimizer=Adam(lr=lr,beta_1=b1))

    # return gan_trainer
    return dm_trainer,gm_trainer#,gan_trainer

print('T...')
dmt,gmt = trainer(gm,gmf,dm,dmf)

fsong = song.astype('float32')
fsong /= 32767.

def r(ep=10):
    for e in range(ep):
        print('ep:',e)
        # randomly sample 32 pair of patches of length 1024
        big_batch_size = 8
        length_dataset = fsong.shape[0]
        indices = np.random.choice(length_dataset-1024*2,big_batch_size,replace=False)

        inputs = np.stack([fsong[indice:indice+1024] for indice in indices])
        outputs = np.stack([fsong[indice+1024:indice+1024*2] for indice in indices])

        dmt.fit([inputs,outputs],
        np.zeros((big_batch_size,1)),
        nb_epoch=1,
        batch_size=8,
        shuffle=False
        )

        gmt.fit([inputs],
        np.zeros((big_batch_size,1)),
        nb_epoch=1,
        batch_size=8,
        shuffle=False
        )
