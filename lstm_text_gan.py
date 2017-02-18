
from __future__ import print_function
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.data_utils import get_file
# from keras import backend as K
import numpy as np
import random
import sys

from tensorflow_boilerplate import *
import canton as ct

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

def dis():
    i = Input((time_steps,8),name='dis_input')
    # input: time_steps chars, each 256 possibility
    # (batch, time_steps, 256)
    inp = i

    # i = Convolution1D(128, 1, activation='tanh',name='dis_embed')(i)
    # # # embed into vector of 4

    gru = GRU(32,'DGRU')
    i = Lambda(lambda i:gru(i))(i)

    # output shape (batch, time_steps, 32)
    i = Lambda(lambda i:tf.reshape(i[:,time_steps-1,:],[-1,32]))(i)
    i = Dense(1,activation='sigmoid')(i)

    model = Model(input=inp,output=i)
    # model.trainable_weights += gru.get_variables()
    return model,gru

def gen():
    i = Input((time_steps,zed),name='gen_input') # some random vector
    inp = i

    gru = GRU(40,'GGRU')
    i = Lambda(lambda i:gru(i))(i)

    # output shape (batch, time_steps, 32)
    # i = Convolution1D(32,1,activation='sigmoid')(i)
    i = Convolution1D(8,1,activation='sigmoid')(i)
    # shape: (batch, time_steps, 256)
    model = Model(input=inp,output=i)
    # model.trainable_weights += gru.get_variables()
    return model,gru

# dm,dgru = dis()
# dm.summary()
# gm,ggru = gen()
# gm.summary()

# pm,pgru = pred()
# gm.compile(loss='mse',optimizer='sgd')
# dm.compile(loss='mse',optimizer='sgd')

# from gan_common import gan
#
# gan_feed = gan(gm,dm,
#     additional_d_weights=dgru.get_variables(),
#     additional_g_weights=ggru.get_variables())

def mymodel_builder():
    can = ct.Can()
    layers = [ct.GRU(256),ct.Dense(256,64),ct.Dense(64,256)]
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

# def mymodel():
#     gru = GRU(256,'mGRU')
#     den = dense(256,64)
#     den2 = dense(64,256)
#     def call(i):
#         nonlocal gru,den
#         i = gru(i)
#         # output shape (batch, time_steps, 128)
#         # i = tf.reshape(i[:,tf.shape(i)[1]-1,:],[-1,64])
#         ms,ls = tf.shape(i)[1],tf.shape(i)[2]
#         i = tf.reshape(i,[-1,ls])
#         i = den(i)
#         i = tf.tanh(i)
#         i = den2(i)
#         i = tf.reshape(i,[-1,ms,256])
#         # i = tf.sigmoid(i)
#         # i = den2(i)
#         # i = tf.nn.softmax(i)
#         # i = tf.nn.softmax(i)
#         return i
#
#     # def longcall(i):
#     #     i = gru(i)
#     #     i = den(i)
#     #     return i
#     return call

mymodel = mymodel_builder()

def feed_gen():
    x = tf.placeholder(tf.float32, shape=[None, None, corpus.shape[1]])
    y = mymodel(x)
    gt = tf.placeholder(tf.float32, shape=[None, None, corpus.shape[1]])
    loss = categorical_cross_entropy(y,gt)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(
        loss,var_list=mymodel.get_weights())

    def feed(minibatch,labels):
        nonlocal train_step,loss,x,gt
        sess = ct.get_session()
        res = sess.run([loss,train_step],feed_dict={x:minibatch,gt:labels})
        return res[0]

    def predict(minibatch):
        nonlocal y,x
        sess = ct.get_session()
        res = sess.run([y],feed_dict={x:minibatch})
        return res[0]

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

def ro(ep=1000):
    sess = K.get_session()
    length = len(corpus)
    batch_size = 128
    mbl = time_steps * batch_size
    sr = length - mbl

    for i in range(ep):
        print('---------------------------')
        print('iter',i)

        # sample from corpus
        j = np.random.choice(sr)

        minibatch = corpus[j:j+mbl]
        minibatch.shape = [batch_size, time_steps, corpus.shape[1]]

        z_input = np.random.normal(loc=0.,scale=1.,
            size=(batch_size,time_steps,zed))

        # dm.reset_states()
        # gm.reset_states()

        # train for one step
        losses = gan_feed(sess,minibatch,z_input)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show()

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

def show():
    z_input = np.random.normal(loc=0.,scale=1.,
        size=(1,time_steps,zed))
    res = gm.predict(z_input)[0]
    sentence = ''
    for i in range(len(res)):
        sentence += chr(eight2char(res[i]))
    print(sentence)
