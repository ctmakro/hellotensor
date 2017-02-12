from __future__ import print_function

import tensorflow as tf

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
# from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import math
import random

import numpy as np

import cv2

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

starry_night = cv2.imread('starry_night.jpg').astype('float32') / 255. - .5
guangzhou = cv2.imread('DSC_0896cs_s.jpg').astype('float32') / 255. - .5

from keras_resnet import cake,neck,relu

def dis():
    i = Input((None,None,3))
    inp = i

    i = Convolution2D(16,3,3,border_mode='same')(i)
    i = neck(16,16,1)(i)
    i = neck(16,16,1)(i)
    i = neck(16,32,2)(i)
    i = neck(32,32,1)(i)
    i = neck(32,32,1)(i)
    i = neck(32,32,2)(i)
    i = bn(i)
    i = relu(i)
    i = GlobalAveragePooling2D()(i)
    i = Dense(1,activation='sigmoid')(i)

    return Model(input=inp,output=i)

dm = dis()
dm.summary()

def into_variable(value):
    v = tf.Variable(initial_value=value)
    sess = K.get_session()
    sess.run([tf.variables_initializer([v])])
    return v

def gan(d):
    # initialize a GAN trainer

    output_size = [256,256]
    # create white_noise_image
    global white_noise_image
    white_noise_image = tf.Variable(
        tf.random_normal([1]+output_size+[3], stddev=1e-22),
        dtype=tf.float32,name='white_noise_image')

    # initialize the white_noise_image
    K.get_session().run([tf.variables_initializer([white_noise_image])])

    sn = starry_night.view()
    sn.shape = (1,)+sn.shape
    real_image = into_variable(sn)

    gscore = d(white_noise_image)
    rscore = d(real_image)

    def gan_loss(gscore,rscore):
        def log_eps(i):
            return K.log(i+1e-11)

        # single side label smoothing: replace 1.0 with 0.9
        dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
        gloss = - K.mean(log_eps(gscore))

        return dloss,gloss

    dloss,gloss = gan_loss(gscore,rscore)

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-3,.9 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    update_wd = optimizer.minimize(dloss,var_list=d.trainable_weights)
    update_wg = optimizer.minimize(gloss,var_list=[white_noise_image])

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = K.learning_phase()

    def gan_feed():
        # actual GAN trainer
        nonlocal train_step,losses,learning_phase
        sess = K.get_session()
        res = sess.run([train_step,losses],feed_dict={
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed

feed = gan(dm)

def r(ep=10,maxlr=.01):
    import time
    t = time.time()
    for i in range(ep):
        t = time.time()
        print('ep',i)
        lr = maxlr #* (math.cos((i%10)*math.pi/9)+1)/2 + 1e-9
        print('lr:',lr)
        # loss = feed(lr=lr)
        loss = feed()
        t = time.time()-t
        print('dloss: {:6.6f}, gloss: {:6.6f}, {:6.4f}/run, {:6.4f}/s'.format(loss[0],loss[1],t,1/t))

        # if i%5==0:
        #     saveit = True if i%20==0 else False #every 100 ep
        #     show(save=saveit)

show_counter = 0
show_prefix = str(np.random.choice(1000))

def show(save=False):
    sess = K.get_session()
    res = sess.run(white_noise_image)
    image = res[0]
    image += 0.5
    cv2.imshow('result',image)
    cv2.waitKey(1)
    cv2.waitKey(1)
    if save:
        global show_counter,show_prefix
        cv2.imwrite('./log/'+show_prefix+'_'+str(show_counter)+'.jpg',image*255.)
        show_counter+=1
    return image
