from __future__ import print_function
import numpy as np

import tensorflow as tf
import canton as ct
from canton import *

def load_dataset():
    from load_drone_data import load_data
    ximages,yvalues = load_data(False)
    ximages = ximages.astype('float32')/255.

    total = ximages.shape[0]
    split = int(total*45/50)

    X_train = ximages[0:split]
    y_train = yvalues[0:split]

    X_test = ximages[split:total]
    y_test = yvalues[split:total]

    print('X_train shape:', X_train.shape)
    print('y_train shape:',y_train.shape)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np.reshape(y_train,y_train.shape+(1,)) # into [NHW1]
    Y_test = np.reshape(y_test,y_test.shape+(1,))
    print('Y_train after conversion:',Y_train.shape)

    return X_train,Y_train,X_test,Y_test

xt,yt,xv,yv = load_dataset()

def detector():
    c = Can()
    c.add(Conv2D(1,16,k=3,usebias=False))
    c.add(ResConv(16,16,std=1))
    c.add(ResConv(16,16,std=1))
    c.add(ResConv(16,32,std=2)) # 32
    c.add(ResConv(32,32,std=1))
    c.add(ResConv(32,32,std=1))
    c.add(ResConv(32,32,std=2)) # 16
    c.add(ResConv(32,32,std=1))
    c.add(ResConv(32,32,std=1))
    c.add(ResConv(32,32,std=2)) # 8
    c.add(ResConv(32,32,std=1))
    c.add(Act('relu'))
    c.add(Conv2D(32,1,k=1,std=1)) # 8
    c.add(Act('sigmoid'))
    c.chain()
    return c

def detector2():
    c = Can()
    c.add(Conv2D(1,32,k=5,std=2)) # 32
    c.add(Act('lrelu'))
    c.add(Conv2D(32,32,k=5,std=2)) # 16
    c.add(Act('lrelu'))
    c.add(Conv2D(32,32,k=5,std=2)) # 8
    c.add(Act('lrelu'))
    c.add(Conv2D(32,1,k=3,std=1)) # 8
    c.add(Act('sigmoid'))
    c.chain()
    return c

tec = detector()
tec.summary()

def trainer():
    x,gt = ct.ph([None,None,1]), ct.ph([None,None,1])
    y = tec(x)
    loss = ct.binary_cross_entropy_loss(y,gt)
    lr = tf.Variable(1e-3)

    print('connecting optimizer...')
    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(loss,var_list=tec.get_weights())

    def feed(xin,yin,ilr):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={x:xin,gt:yin,lr:ilr})
        return res[1] # loss
    return feed

feed = trainer()
ct.get_session().run(ct.gvi()) # global init

def r(ep=10,lr=1e-3):
    for i in range(ep):
        print('ep',i)
        bs = 20
        for j in range(len(xt)//bs):
            mbx = xt[j*bs:(j+1)*bs]
            mby = yt[j*bs:(j+1)*bs]
            loss = feed(mbx,mby,lr)
            print(j*bs,'loss:',loss)

def show(): # evaluate result on validation set
    from vis import autoscaler,batch_image_array
    import cv2

    indices = np.random.choice(len(yv), 16)
    mbx = np.take(xv, indices, axis=0)
    mby = np.take(yv, indices, axis=0)
    mby += 0.1 # gray tint
    result = tec.infer(mbx)

    wall1,s1 = batch_image_array(mbx)
    wall2,s2 = batch_image_array(result)
    wall3,s3 = batch_image_array(mby)

    cv2.imshow('valset'+str(s1),wall1)
    cv2.imshow('vallbl'+str(s3),wall3)
    cv2.imshow('valres'+str(s2),wall2)
    cv2.waitKey(1)
    cv2.waitKey(1)
