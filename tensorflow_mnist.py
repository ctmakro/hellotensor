from __future__ import print_function

from tensorflow_boilerplate import *
from tensorflow_session import *
import tensorflow as tf
import numpy as np

class mynet:
    def __init__(self):
        self.name = 'mynet'

    def summary(self):
        summary_scope(self.name)

    def model(self,inp):
        with tf.variable_scope(self.name):
            print('building model...')

            i = inp
            i = tf.reshape(i,[-1,28,28,1]) # reshape into 4d tensor

            i = conv2d(1,16,3)(i)

            # i = resconv(i,16,16)
            i = resconv(i,16,16)
            i = resconv(i,16,16,std=2)

            # i = resconv(i,16,16)
            i = resconv(i,16,16)
            i = resconv(i,16,32,std=2)

            # i = resconv(i,32,32)
            i = resconv(i,32,32)
            i = resconv(i,32,64,std=2)

            i = bn(i)
            i = relu(i)
            i = conv2d(64,10,1)(i)

            i = tf.reduce_mean(i,[1,2]) # 2d tensor (N, onehot)

            out = i
            print('model built.')
            return out

net = mynet()
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y = net.model(x,'dis') # use same name to share params.
#
# gt = tf.placeholder(tf.float32, shape=[None, 10])
#
# mr = ModelRunner(inputs=x,outputs=y,gt=gt)
# mr.set_loss(categorical_cross_entropy(y,gt)) # loss(y,gt)
# mr.set_optimizer(Adam(1e-3))
# mr.set_acc(batch_accuracy(y,gt)) # optional. acc(y,gt)

# f = make_function(x,y)

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

# summary('dis')

# r = mr.get_epoch_runner(xtrain,ytrain,xtest,ytest)
# r = mr.defaultEpochRunner(xtrain,ytrain)

amr = AdvancedModelRunner()
amr.net = net
amr.optimizer = Adam(1e-3)
amr.epoch_runner_preload(xtrain,ytrain)
