
from __future__ import print_function
from keras.utils.data_utils import get_file
# from keras import backend as K
import numpy as np
import random
import sys

# from tensorflow_boilerplate import *
import canton as ct
import tensorflow as tf

zed = 32
time_steps = 1024

def char2eight(n):
    binary = "{0:b}".format(n)
    rep = np.zeros((8,),dtype='float32')
    for i in range(len(binary)): # 0..2
        rep[i+8-len(binary)] = 1 if binary[i] == '1' else 0
    return rep

def eight2char(e):
    out = 0
    for i in range(len(e)):
        out += (1 if e[7-i]>0.5 else 0) * 2**i
    return int(out)

def one_hot(tensor,dims):
    onehot = np.zeros((len(tensor),dims),dtype='float32')
    for i in range(dims):
        onehot[...,i] = tensor[...] == i
    return onehot

def to_asc(text):
    asc = np.zeros((len(text),),dtype='uint8')
    # convert into ascii
    for i in range(len(text)):
        asc[i] = ord(text[i])
    return asc

def textdata():
    path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    text = open(path).read()
    length = len(text)
    print('got corpus length:', length)

    # convert into ascii
    asc = to_asc(text)

    # convert into 8dim
    # for i in range(length):
    #     onehot[i] = char2eight(asc[i])

    # convert into onehot
    onehot = one_hot(asc,256)

    return text,onehot

text,corpus = textdata()
print('corpus loaded. corpus[0]:',corpus[0])

def mymodel_builder():
    can = ct.Can()
    layers = [ct.GRU(256,256),ct.Dense(256,64),ct.Dense(64,256)]
    can.incan(layers)
    def call(i):
        i = layers[0](i)
        # (batch, time_steps, 256)
        shape = tf.shape(i)
        b,t,d = shape[0],shape[1],shape[2]

        i = tf.reshape(i,[-1,d])

        i = layers[1](i)
        i = layers[2](i)

        i = tf.reshape(i,[-1,t,d])
        return i
    can.set_function(call)
    return can

mymodel = mymodel_builder()

def feed_gen():
    x = tf.placeholder(tf.float32, shape=[None, None, corpus.shape[1]])
    y = mymodel(x)
    gt = tf.placeholder(tf.float32, shape=[None, None, corpus.shape[1]])
    loss = ct.mean_softmax_cross_entropy(y,gt)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(
        loss,var_list=mymodel.get_weights())

    def feed(minibatch,labels):
        nonlocal train_step,loss,x,gt
        sess = ct.get_session()
        res = sess.run([loss,train_step],feed_dict={x:minibatch,gt:labels})
        return res[0]

    def predict(minibatch):
        return mymodel.infer(minibatch)

    return feed,predict

feed,predict = feed_gen()

sess = ct.get_session()
sess.run(tf.variables_initializer(mymodel.get_weights()))
sess.run(tf.global_variables_initializer())

def r(ep=100):
    length = len(corpus)
    batch_size = 1
    mbl = time_steps * batch_size
    sr = length - mbl - time_steps - 2
    for i in range(ep):
        print('---------------------iter',i,'/',ep)

        j = np.random.choice(sr)

        minibatch = corpus[j:j+mbl]
        minibatch.shape = [batch_size, time_steps, corpus.shape[1]]
        labels = corpus[j+1:j+mbl+1]
        labels.shape = [batch_size,time_steps,corpus.shape[1]]

        loss = feed(minibatch,labels)
        print('loss:',loss)

        if i%100==0 : pass#show2()

def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

import sys
def show2():
    t = 'the'
    for i in range(400):
        # convert into ascii
        asc = to_asc(t[-time_steps:])
        hot = one_hot(asc,256)
        hot.shape = (1,)+hot.shape
        #print(hot.shape)

        res = predict(hot)[0]
        dist = softmax(res[-1])

        code = np.random.choice(256, p=dist)
        # code = np.argmax(dist)

        #print(code)
        char = chr(code)
        t+=char

        sys.stdout.write( '%s' % char )
        sys.stdout.flush()
    print('')
