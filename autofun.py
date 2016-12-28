from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D
from keras.optimizers import SGD, Adadelta, Adam, RMSprop

from keras.utils import np_utils
import keras
import math
import keras.backend as K
import numpy as np

aa = keras.layers.advanced_activations

def leaky(alpha=0.3):
    return aa.LeakyReLU(alpha=alpha)

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import sgdr

print('import complete.')

model = Sequential()

model.add(Dense(32,input_shape=(8,)))
model.add(Activation('relu'))
model.add(Dense(32))
# model.add(leaky())
model.add(Activation('relu'))

model.add(Dense(4))

model.add(Dense(32))
# model.add(leaky())
model.add(Activation('relu'))
model.add(Dense(32))
# model.add(leaky())
model.add(Activation('relu'))

model.add(Dense(8))

model.summary()

# let's train the model using SGD + momentum (how original).
# opt = SGD(lr=0.03, decay=1e-6, momentum=0.95, nesterov=True)

# opt = RMSprop(lr=0.03)
opt = Adam()
# opt= Adadelta()
model.compile(loss='mse',
              optimizer=opt)
            #   metrics=['accuracy'])

num_datapoints = 100000
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

import time

losshist = {'l':[],'vl':[]}

Cb=keras.callbacks.Callback

update_timer = time.time()
class Mycb(Cb):
    def __init__(self):
        super(Cb, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
        lr=K.get_value(self.model.optimizer.lr)

        losshist['l'].append(logs['loss'])
        losshist['vl'].append(logs['val_loss'])

        logs['lr'] = lr
        global update_timer
        if time.time() > update_timer + 3:
            update_timer = time.time()
            updateplot(losshist)

def myfit(mdl,xt,yt,bs,ep,vd):
    return mdl.fit(xt, yt,
              batch_size=bs,
              nb_epoch=ep,
              validation_data=vd,
              shuffle=False,
              callbacks=[
            #   scheduler,
              Mycb()
              ])

def r(ep=30,bs=10000,maxlr=0.05,minlr=0.0001):
    global scheduler
    scheduler = sgdr.gen_scheduler(maxlr=maxlr,minlr=minlr,t0=10,tm=1)

    startanimplot()
    thishist = myfit(model,X_train,Y_train,
        bs=bs,ep=ep,vd=(X_test,Y_test))

    return thishist

def startanimplot():
    global fig
    global ax

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

def updateplot(losshist):
    global fig
    global ax

    ax.clear()
    ax.grid(True)
    ax.set_yscale('log')
    for name in losshist:
        ax.plot(losshist[name],label=name)

    ax.legend()
    #plt.draw()
    fig.canvas.draw()
    plt.pause(0.001)

def plothist(h):
    plt.yscale('log')
    plt.title('loss')
    plt.grid(True)
    plt.plot(h.history['loss'],label='loss')
    # plt.plot(h.history['val_loss'],label='val_loss')
    plt.legend()
    plt.show(block=False)

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
