import tensorflow as tf
import canton as ct
from canton import *
from cv2tools import vis

import numpy as np
import cv2

from lets_gan_canton import gm

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

ct.get_session().run(tf.global_variables_initializer())
gm.load_weights('gm_ls_new.npy')

def imagine(dim=4,save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(1,dim,dim,zed))
    gened = gm.infer(i)

    gened *= 0.5
    gened +=0.5
    im=gened[0]

    vis.show_autoscaled(im,name='imagined',limit=800.)
    if save!=False:
        cv2.imwrite(save,im*255)
