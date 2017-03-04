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
    
    #c.add(deconv(ngf*4,ngf*4,upscale=1))
    
    c.add(deconv(ngf*4,ngf*2))
    
    #c.add(deconv(ngf*2,ngf*2,upscale=1))
    
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
        
    def batch_disc(i):
        #assume i shape [N H W C]
        s = tf.shape(i)
        NHWC1 = tf.expand_dims(i,4)
        AHWCN = tf.expand_dims(tf.transpose(i,[1,2,3,0]),0)
        diffs = NHWC1 - AHWCN # [N H W C N]
        abs_diffs = tf.abs(diffs)
        # shape [N H W C N]
        feat = tf.reduce_mean(tf.exp(-abs_diffs), [3,4])#[N H W]
        feat = tf.expand_dims(feat,3)
        # shape [N H W 1]
        out = tf.concat([i, feat],axis=-1) # [N H W C+1]
        return out
        
        #http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
    
    cd = Can()
    cd.set_function(batch_disc)

    def conv(nip,nop,usebn=True,std=2):
        cv = Can()
        cv.add(Conv2D(nip,nop,k=4,std=std,usebias=False))
        if usebn:
            cv.add(BatchNorm(nop))
        cv.add(Act('lrelu'))
        cv.add(cd)
        cv.chain()
        return cv

    ndf = 32
    c.add(conv(3,ndf*1,usebn=False)) # 16
    c.add(conv(ndf*1+1,ndf*2))
    c.add(conv(ndf*2+1,ndf*4))
    c.add(conv(ndf*4+1,ndf*8)) # 2
    c.add(Conv2D(ndf*8+1,1,k=2,padding='VALID'))
    c.add(Reshape([1]))
    #c.add(Act('sigmoid'))
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

    noise = tf.random_normal(mean=0.,stddev=1.,shape=[batch_size,1,1,zed])
    real_data = ct.ph([None,None,3])
    inl = tf.Variable(1e-11)
    
    def noisy(i):
        return i + tf.random_normal(mean=0,stddev=inl,shape=tf.shape(i))

    generated = g(noise)
    
    gscore = d(noisy(generated))
    rscore = d(noisy(real_data))
    
    def log_eps(i):
        return tf.reduce_mean(tf.log(i+1e-8))

    # single side label smoothing: replace 1.0 with 0.9
    #dloss = - (log_eps(1-gscore) + .1 * log_eps(1-rscore)+ .9*log_eps(rscore))
    #gloss = - log_eps(gscore)
    
    dloss = tf.reduce_mean((gscore-0)**2 + (rscore-1)**2)
    gloss = tf.reduce_mean((gscore-1)**2)

    Adam = tf.train.AdamOptimizer
    #Adam = tf.train.MomentumOptimizer
    
    lr,b1 = tf.Variable(1.2e-4),.5 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)
    #optimizer = Adam(lr)

    def l2(m):
        l = m.get_weights()
        return tf.reduce_mean([tf.reduce_mean(i**2)*0.01 for i in l])
    update_wd = optimizer.minimize(dloss,var_list=d.get_weights())
    update_wg = optimizer.minimize(gloss,var_list=g.get_weights())

    train_step = [update_wd, update_wg]
    losses = [dloss,gloss]


    def gan_feed(sess,batch_image,nl,lllr):
        # actual GAN training function
        nonlocal train_step,losses,noise,real_data

        res = sess.run([train_step,losses],feed_dict={
        real_data:batch_image,
        inl:nl,lr:lllr,
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]
    return gan_feed

print('generating GAN...')
gan_feed = gan(gm,dm)

ct.get_session().run(tf.global_variables_initializer())
print('Ready. enter r() to train')

noise_level=.1
def r(ep=10000,lr=1e-4):
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

        # train for one step
        losses = gan_feed(sess,minibatch,noise_level,lr)
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

    patches = num
    height = max(1,int(math.sqrt(patches)*0.9))
    width = int(patches/height+1)

    img = np.zeros((height*(uh+1), width*(uw+1), 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index<num:
                channels = arr[index]
                img[row*(uh+1):row*(uh+1)+uh,col*(uw+1):col*(uw+1)+uw,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale

def show(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(1,8,8,zed))
    gened = gm.infer(i)

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite(save,im*255)
