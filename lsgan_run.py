import tensorflow as tf
import canton as ct
from canton import *

import numpy as np
import cv2

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

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

gm = gen_gen()  
ct.get_session().run(tf.global_variables_initializer())

gm.load_weights('gm.h5')

def imagine(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(1,32,32,zed))
    gened = gm.infer(i)

    gened *= 0.5
    gened +=0.5
    im=gened[0]
    
    cv2.imshow('show',im)
    cv2.waitKey(1)
    if save!=False:
        cv2.imwrite(save,im*255)