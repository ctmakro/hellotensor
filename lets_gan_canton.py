from __future__ import print_function

import tensorflow as tf
import canton as ct
from canton import *
from keras.datasets import cifar10

# from keras.optimizers import *
from keras.utils import np_utils

import math

import numpy as np
import cv2

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

def cifar():
    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    X_train-=0.5
    X_test-=0.5

    return X_train,Y_train,X_test,Y_test
print('loading cifar...')
xt,yt,xv,yv = cifar()

def gen_gen():
    c = Can()
    c.add(Reshape([1,1,zed]))

    def deconv(nip,nop,tail=True,upscale=2):
        dc = Can()
        dc.add(Up2D(upscale))
        dc.add(Conv2D(nip,nop,k=4,std=1,usebias=not tail))
        if tail:
            dc.add(BatchNorm(nop))
            dc.add(Act('relu'))
        dc.chain()
        return dc

    ngf = 32
    c.add(deconv(zed,ngf*8,upscale=4)) #4
    c.add(deconv(ngf*8,ngf*4))
    c.add(deconv(ngf*4,ngf*2))
    c.add(deconv(ngf*2,ngf*1)) #32
    c.add(deconv(ngf*1,3,tail=False,upscale=1))
    c.add(Act('tanh'))
    c.chain()
    return c

def dis_gen():
    c = Can()

    def concat_diff(x): # batch discrimination - increase generation diversity.
        avg = tf.reduce_mean(x,axis=0) # average color of this batch

        l1 = abs(x[:] - avg) # l1 distance of each image to average color

        avgl1 = tf.reduce_mean(l1,axis=-1,keep_dims=True)
        # average l1d as new channel: shape [N H W 1]

        out = tf.concat([x,avgl1], axis=-1) # shape [N H W C+1]
        return out
    cd = Can()
    cd.set_function(concat_diff)

    def conv(nip,nop,usebn=True,std=2):
        cv = Can()
        cv.add(Conv2D(nip,nop,k=4,std=std,usebias=False))
        if usebn:
            cv.add(BatchNorm(nop))
        cv.add(Act('elu'))
        cv.add(cd)
        cv.chain()
        return cv

    ndf = 40
    c.add(conv(3,ndf*1,usebn=False)) # 16
    c.add(conv(ndf*1+1,ndf*2))
    c.add(conv(ndf*2+1,ndf*4))
    c.add(conv(ndf*4+1,ndf*8)) # 2

    c.add(Conv2D(ndf*8+1,1,k=2,padding='VALID'))
    c.add(Reshape([1]))
    c.add(Act('sigmoid'))
    c.chain()
    return c

print('generating G...')
gm = gen_gen()
gm.summary()

print('generating D...')
dm = dis_gen()
dm.summary()

def gan(g,d):
    # initialize a GAN trainer
    # this is the fastest way to train a GAN in TensorFlow
    # two models are updated simutaneously in one pass

    noise = tf.random_normal(mean=0.,stddev=1.,shape=[batch_size, zed])
    real_data = ct.ph([None,None,3])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return tf.log(i+1e-13)

    # single side label smoothing: replace 1.0 with 0.95
    dloss = - tf.reduce_mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .95 * log_eps(rscore))
    gloss = - tf.reduce_mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    update_wd = optimizer.minimize(dloss,var_list=d.get_weights())
    update_wg = optimizer.minimize(gloss,var_list=g.get_weights())

    train_step = [update_wd, update_wg]
    losses = [dloss,gloss]


    def gan_feed(sess,batch_image):
        # actual GAN training function
        nonlocal train_step,losses,noise,real_data

        res = sess.run([train_step,losses],feed_dict={
        real_data:batch_image,
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]
    return gan_feed

print('generating GAN...')
gan_feed = gan(gm,dm)

ct.get_session().run(tf.global_variables_initializer())
print('Ready. enter r() to train')

noise_level=.1
def r(ep=10000):
    sess = ct.get_session()

    np.random.shuffle(xt)
    shuffled_cifar = xt
    length = len(shuffled_cifar)

    for i in range(ep):
        global noise_level
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from cifar
        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
        minibatch += np.random.normal(loc=0.,scale=noise_level,size=minibatch.shape)

        # train for one step
        losses = gan_feed(sess,minibatch)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 20==0: show()

def autoscaler(img):
    limit = 400.
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break

    img = cv2.resize(img,dsize=(int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = int(num+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale

def show(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(64,zed))
    gened = gm.infer(i)

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite(save,im*255)
