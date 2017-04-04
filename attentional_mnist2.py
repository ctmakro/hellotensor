from __future__ import print_function

import tensorflow as tf
import numpy as np
import canton as ct
from canton import *

def mnist_data():
    print('loading mnist...')
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # reshape into 2d
    # X_train = X_train.reshape(X_train.shape[0],784)
    # X_test = X_test.reshape(X_test.shape[0],784)

    X_train.shape += 1,
    X_test.shape += 1,

    print('X_train shape:', X_train.shape, X_train.dtype)
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

xt,yt,xv,yv = mnist_data()

gg2d = GRU_Glimpse2D(num_h=64, num_receptors=9, channels=1, pixel_span=28)

def classifier():
    c = Can()
    g = c.add(gg2d)
    d1 = c.add(TimeDistributedDense(64,10))
    c.unit = g.unit # rnn onepass instance
    def call(i,return_hidden_states=False):
        # input should be image sequence: [NTHWC]
        hidden_states = g(i,state_shaper=lambda inp,hid:[tf.shape(inp)[0],hid])
        # state_shaper is used to determine the shape of the initial state
        # for rnn
        # i -> [batch, timesteps, hidden]

        i = d1(hidden_states)
        if not get_training_state():
            i = Act('softmax')(i) # apply softmax only while not training

        if return_hidden_states:
            return i, hidden_states
        else:
            return i
    c.set_function(call)
    return c

gg2dclf = classifier()
gg2dclf.summary()

def trainer():
    inp = ct.ph([None,None,1]) # image
    gt = ct.ph([10]) # labels

    x = inp-0.5
    x = tf.expand_dims(x,axis=1) #[NHWC] -> [N1HWC]
    gt2 = tf.expand_dims(gt,axis=1) #[batch, dims] -> [batch, 1, dims]
    timesteps = 8 # how many timesteps would you evaluate the RNN

    x = tf.tile(x,multiples=[1,timesteps,1,1,1])
    gt2 = tf.tile(gt2,multiples=[1,timesteps,1])

    y = gg2dclf(x) # [batch, timesteps, 10]

    loss = mean_softmax_cross_entropy(y, gt2)
    # mean of cross entropy, over all timesteps.

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss, var_list=gg2dclf.get_weights())

    def feed(img,lbl):
        sess = get_session()
        res = sess.run([train_step,loss],feed_dict={
            inp:img, gt:lbl
        })
        return res[1]

    # extract foveal pattern from hidden states
    set_training_state(False) # set training state to false to enable softmax
    y_softmaxed,hiddens = gg2dclf(x,return_hidden_states=True)
    # [batch, timesteps, 10], [batch, num_h]
    set_training_state(True)

    hs = tf.shape(hiddens)
    hiddens = tf.reshape(hiddens,shape=[-1,hs[2]]) #[batch*time, dims]

    offsets = gg2dclf.unit.get_offset(hiddens)

    shifted_means = gg2dclf.unit.glimpse2d.shifted_means_given_offsets(offsets)
    shifted_means = tf.reshape(shifted_means,shape=[hs[0],hs[1],-1,2]) #[batch*time,num_receptors,2] -> [batch,time,num_receptors,2]

    variances = gg2dclf.unit.glimpse2d.variances() #[num_receptors,1]

    def test(img):
        sess = get_session()
        res = sess.run([x,y_softmaxed,shifted_means,variances],feed_dict={
            inp:img
        })
        return res

    return feed,test

feed,test = trainer()

get_session().run(gvi())

def r(ep=10):
    length = len(xt)
    bs = 20
    for i in range(ep):
        print('ep:',i)
        for j in range(0,length,bs):
            mbx = xt[j:j+bs]
            mby = yt[j:j+bs]
            loss = feed(mbx,mby)
            print(j,'loss:',loss)
            if j% 200 == 0:
                show()

def show():
    import cv2
    from cv2tools import vis,filt

    index = np.random.choice(len(xt))
    img = xt[index:index+1]
    tiledx,y_softmaxed,shifted_means,variances = test(img)

    for idxt, dist in enumerate(y_softmaxed[0]):
        print('step',idxt,'guess:',np.argmax(dist))

    tiledx += 0.5
    tiledx_copy = tiledx.copy()
    tiledx = (tiledx*255.).astype('uint16') # 16-bit-ify
    tiledx = np.tile(tiledx,(1,1,1,3)) # colorify

    shifted_means += np.array([img.shape[1]-1,img.shape[2]-1],dtype='float32')/2
    # shift from image center to image coordinates

    # draw the circles...
    for idxt,receptors in enumerate(shifted_means[0]):
        tmp = tiledx[0,idxt]*0 # [HWC]
        for idxr,receptor in enumerate(receptors):
            tmp += cv2.circle(
                np.zeros_like(tmp,dtype='uint8'),
                (int(receptor[1]*16), int(receptor[0]*16)),
                radius=int(np.sqrt(variances[idxr,0])*16),
                color=(80,140,180), thickness=-1,
                lineType=cv2.LINE_AA, shift=4)

        tiledx[0,idxt] = tiledx[0,idxt]*0.5 + tmp*0.5

    tiledx = tiledx.clip(min=0,max=255).astype('uint8')
    vis.show_batch_autoscaled(tiledx_copy[0],name='input sequence')
    vis.show_batch_autoscaled(tiledx[0],name='attention over time')
