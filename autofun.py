from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, Adadelta, Adam, RMSprop

from keras.utils import np_utils
import keras
import math
import keras.backend as K
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import sgdr

print('import complete.')

def act(i):
    return Activation('elu')(i)

def cake(i):
    i = Dense(8)(i)
    i = act(i)
    i = Dense(16)(i)
    i = act(i)
    i = Dense(8)(i)
    i = act(i)
    return i

inp = Input(shape=(8,))

i = cake(inp)
i = Dense(4)(i)
i = cake(i)

out = Dense(8)(i)

model = Model(input=inp,output=out)

model.summary()

model.compile(loss='mse',
              optimizer=Adam())

num_datapoints = 10000
num_train = num_datapoints*95/100

# generate 8-d redundant data
Xs = np.random.rand(num_datapoints,8)*2
Xs[:,2]=Xs[:,0]*Xs[:,1]*10+Xs[:,6]*1
Xs[:,3]=Xs[:,4]*Xs[:,6]*5+Xs[:,0]*2
Xs[:,5]=Xs[:,6]*Xs[:,1]*2+Xs[:,4]*5
Xs[:,7]=Xs[:,4]*Xs[:,1]*1+Xs[:,6]*10

# 2357 are generated, 0146 are independent

# data normalization
for i in range(Xs.shape[1]):
    upper = max(Xs[:,i])
    lower = min(Xs[:,i])
    print('Xs dim {} : from {} to {}'.format(i,lower,upper))
    Xs[:,i]-=(upper+lower)/2 # mean 0
    Xs[:,i]/=(upper-lower) # range 1

X_train = Xs[0:num_train,:]
X_test = Xs[num_train:num_datapoints,:]
Y_train = X_train
Y_test = X_test
import histlogger as hl
lc = hl.LoggerCallback(keys=[{'loss':[],'val_loss':[]}],interval=1.5)

def r(ep=30,bs=10000,maxlr=0.05,minlr=0.0001):
    model.fit(X_train, Y_train,
              batch_size=bs,
              nb_epoch=ep,
              validation_data=(X_test,Y_test),
              shuffle=False,
              callbacks=[
              lc
              ])

def test(inst=10):
    import random
    range_start = len(X_test)-inst-1
    start = int(random.random()*range_start)
    i = X_test[start:start+inst]
    p = model.predict(i)
    for idx,item in enumerate(i):
        print('---------------')
        print('orig:',i[idx])
        print('pred:',p[idx])
