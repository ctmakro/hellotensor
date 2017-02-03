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

def disp(img):
    import vis
    vis.autoscaler_show(img)

def loopone(tensor):
    for j in range(tensor.shape[0]):
        disp(tensor[j])
    cv2.destroyAllWindows()

EXAMPLE_PATH = 'd:/stage1/'

def loadone(filename):
    fullpath = EXAMPLE_PATH + filename
    npz = np.load(fullpath)
    t,h = npz['tensors'],npz['hashes']
    return t,h

# t1,h1 = loadone('0c60f4b87afcb3e2dfa65abbbf3ef2f9.npz')

# loopone(t1)

def relu(i):
    return Activation('relu')(i)

def build_model():
    inp = Input(shape=(None,None,None,1)) #5d tensor
    # inp = Input(shape=(150,150,150,1)) #5d tensor
    i = inp

    def res3(i,nip,nop,std=1): # 3d resnet
        ident = i

        nop4 = int(nop/4)

        i = Convolution3D(nop4,1,1,1,subsample=(std,std,std))(i)
        i = relu(i)
        i = Convolution3D(nop4,3,3,3,border_mode='same')(i)
        i = relu(i)
        i = Convolution3D(nop,1,1,1)(i)
        i = relu(i)

        if(nip==nop and std==1):
            ident = ident
        else:
            ident = Convolution3D(nop,1,1,1,subsample=(std,std,std))(ident)

        i = merge([ident,i],mode='sum')
        return i

    i = Convolution3D(16,3,3,3,border_mode='same')(i)

    i = res3(i,16,16)
    i = res3(i,16,16)
    i = res3(i,16,16,std=2)

    i = res3(i,16,16)
    i = res3(i,16,16)
    i = res3(i,16,16,std=2)

    i = res3(i,16,16)
    i = res3(i,16,16)
    i = res3(i,16,16,std=2)

    i = res3(i,16,16)
    i = res3(i,16,16)
    i = res3(i,16,16,std=2)

    # 5d tensor: batch, slice, width, height, feature

    i = Lambda(lambda x: K.mean(x,axis=(1,2,3)))(i)

    i = Dense(3)(i)
    i = Activation('tanh')(i)
    i = Dense(1)(i)
    i = Activation('sigmoid')(i) # prob

    model = Model(input=inp,output=i)
    return model

model = build_model()
model.summary()

def loadthese(hashes):
    directory = EXAMPLE_PATH
    # subdirs = os.listdir(directory)
    subdirs = [h + '.npz' for h in hashes]
    print('got',len(subdirs),'to load...')
    print('trying to load em all...')
    examples = [loadone(fname) for fname in subdirs]
    print('done')

    tensors = [e[0].reshape((1,)+e[0].shape) for e in examples] # add 1 dim
    # hashes = [e[1] for e in examples]

    # print('hashes are:')
    # print(hashes)

    # return tensors,hashes
    return tensors

def loadlabels():
    import csv
    hashes = []
    labels = []
    with open('d:/stage1_labels/stage1_labels.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0]=='id':
                print('got first row')
                continue
            else:
                hashes.append(row[0])
                labels.append(int(row[1]))
        print('csv done.')

    return hashes,labels

training_hashes,training_labels = loadlabels()

model.compile(loss='binary_crossentropy',optimizer='adam')

def r(ep=10):
    total = len(training_hashes)
    cntr = 0

    for i in range(ep):
        print('ep',i)

        for j in range(1,total,5):
            print('loading',j,j+5)
            this_batch_training_data = loadthese(training_hashes[j:j+5])
            this_batch_training_data = [(td/127.5 - 1.).astype('float32') for td in this_batch_training_data]
            # into float32

            this_batch_labels = np.array(training_labels[j:j+5],dtype='float32').reshape(5,1) # well well well

            print('loaded.')

            for k in range(len(this_batch_training_data)):
                model.fit(this_batch_training_data[k],
                this_batch_labels[k],
                nb_epoch=1,
                batch_size=1,
                shuffle=False
                )
                cntr+=1

                print(cntr,'patient scanned...')
