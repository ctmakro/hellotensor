from __future__ import print_function

# from tensorflow_boilerplate import *
# from tensorflow_session import *
import tensorflow as tf
import numpy as np
import canton as ct
from canton import *

def mycan_build():
    c = Can()
    c.add(Lambda(lambda i:tf.reshape(i,[-1,28,28,1])))
    c.add(Conv2D(1,16,3))

    c.add(ResConv(16,16))
    c.add(ResConv(16,16))
    c.add(ResConv(16,32,std=2))

    c.add(ResConv(32,32))
    c.add(ResConv(32,32))
    c.add(ResConv(32,64,std=2))

    c.add(ResConv(64,64))
    c.add(ResConv(64,64))
    c.add(ResConv(64,64,std=2))

    c.add(Act('relu'))
    c.add(Conv2D(64,10,1))

    c.add(Lambda(lambda i:tf.reduce_mean(i,[1,2])))
    c.chain()
    return c

def mnist_data():
    print('loading mnist...')
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # reshape into 2d
    X_train = X_train.reshape(X_train.shape[0],784)
    X_test = X_test.reshape(X_test.shape[0],784)

    print('X_train shape:', X_train.shape,X_train.dtype)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    def categorical(tensor,cat):
        newshape = tuple(tensor.shape[0:1])+(cat,)
        print(newshape)
        new = np.zeros(newshape,dtype='float32')
        for i in range(cat):
            new[:,i] = tensor[:] == i
        return new

    Y_train = categorical(y_train,10)
    Y_test = categorical(y_test,10)

    print('Y_train shape:',Y_train.shape)
    print(Y_train[0])

    print('mnist loaded.')
    return X_train,Y_train,X_test,Y_test

xtrain,ytrain,xtest,ytest = mnist_data()

mycan = mycan_build()

def feed_gen():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = mycan(x)
    gt = tf.placeholder(tf.float32,shape=[None,10])

    loss = ct.mean_softmax_cross_entropy(y,gt)
    acc = ct.one_hot_accuracy(y,gt)

    optimizer = tf.train.AdamOptimizer(1e-3)
    train_step = optimizer.minimize(loss,var_list=mycan.get_weights())

    def feed(xi,yi,train=True):
        sess = ct.get_session()
        if train:
            res = sess.run([loss,acc,train_step],feed_dict={x:xi,gt:yi})
        else:
            res = sess.run([loss,acc],feed_dict={x:xi,gt:yi})
        return res[0:2]
    return feed

feed = feed_gen()

ct.get_session().run(tf.global_variables_initializer())

def r(ep=10):
    bs = 32
    for i in range(ep):
        print('ep',i)
        for j in range(0,len(xtrain),bs):
            loss,acc = feed(xtrain[j:j+bs],ytrain[j:j+bs])
            print('loss:',loss,'acc:',acc)
        loss,acc = feed(xtest,ytest,train=False)
        print('test loss:',loss,'acc:',acc)
