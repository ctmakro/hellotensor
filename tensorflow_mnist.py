from __future__ import print_function

from tensorflow_boilerplate import *
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

def build_model():
    inp = tf.placeholder(tf.float32, shape=[None, 784])

    i = inp
    i = tf.reshape(i,[-1,28,28,1]) # reshape into 4d tensor

    i = conv2d(1,8,3)(i)

    i = resconv(i,8,8)
    i = resconv(i,8,16,std=2)

    i = resconv(i,16,16)
    i = resconv(i,16,32,std=2)

    i = resconv(i,32,32)
    i = resconv(i,32,64,std=2)

    i = resconv(i,64,10)

    i = tf.reduce_mean(i,[1,2]) # 2d tensor (N, onehot)

    out = i
    return inp,out

x,y = build_model()

gt = tf.placeholder(tf.float32, shape=[None, 10])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, gt))
# loss function: softmax + cross_entropy
acc = batch_accuracy(y,gt)

adam = Adam(1e-3)
train_step = TrainWith(loss,adam)

sess.run(tf.global_variables_initializer())

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

    print('X_train shape:', X_train.shape)
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

    return X_train,Y_train,X_test,Y_test

xtrain,ytrain,xtest,ytest = mnist_data()

summary()

def r(ep=100,bs=128):
    timer = TrainTimer()

    length = len(xtrain) # length of one epoch run
    batch_size = bs

    for i in range(ep):
        timer.epstart()

        print('--------------')
        print('Epoch',i)

        bar = StatusBar(length)

        for j in range(0, length, batch_size):
            # train one minibatch
            inputs = xtrain[j:j+batch_size]
            labels = ytrain[j:j+batch_size]
            res = sess.run([train_step,loss,acc],feed_dict={x:inputs, gt:labels})
            # train_step.run(feed_dict={x:inputs, gt:labels})

            bar.update(min(j+batch_size,length),
            loss=res[1],
            acc =res[2]
            )

        print('')
        timer.epend_report(length)

        # test set
        timer.epstart()
        res = sess.run([loss,acc],feed_dict={x:xtest, gt:ytest})
        duration = timer.epend()
        print('test done in {:6.2f}s loss:{:6.4f} acc:{:6.4f}'.format(duration,res[0],res[1]))
