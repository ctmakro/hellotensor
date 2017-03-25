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

sess = ct.get_session()
sess.run(tf.global_variables_initializer())
gm.load_weights('gm_ls_new.npy')

def imagine2(dim,code,save=False,visualize=True):
    # set the shape
    shape = (1,dim,dim,zed)

    code1 = code[0:zed]
    code2 = code[zed:zed*2]

    # overall tone
    means = code1 - 0.5 # -0.5 or 0.5
    variances = code2 * .3 + .2 # .2 or .5

    # final latent
    z = np.random.normal(loc=means,scale=variances,size=shape)

    gened = gm.infer(z)
    gened *= 0.5 # normalize to 0..1
    gened +=0.5
    im=gened[0]

    if visualize:
        # visualize
        vis.show_autoscaled(im,name='imagined',limit=600.)
        fmap = np.transpose(z,(3,1,2,0)) + .5
        vis.show_batch_autoscaled(fmap,name='z_map',limit=600.)

    # outputs the image
    return im

def codegen(length):
    return np.random.binomial(n=1,p=.5,size=length)

def hex2code(hexstr):
    res = np.zeros((len(hexstr)*4,),dtype='uint8')
    for i in range(len(hexstr)):
        n = int(hexstr[i],16)
        binstring = "{0:b}".format(n)
        for j in range(len(binstring)):
            res[i*4+j] = int(binstring[j])
    return res

def code2hex(code):
    hexstr = ''
    for i in range(0,len(code),4):
        binstr = ''
        for j in range(4):
            binstr += str(code[i+j])
        n = int(binstr,2)
        hexstr_ = hex(n)[2:]
        hexstr += hexstr_
    return hexstr

def imagine3(hexcode=None):
    if hexcode is None:
        code = codegen(length=zed*2)
        hexcode = code2hex(code)
    else:
        code = hex2code(hexcode)

    image = imagine2(dim=8,code=code,visualize=False)
    return image, hexcode

import time
from collections import deque
import threading as th

# use this class if you want something:
# - continously running in the background
# - that exits when program exits
class Waiter:
    def __init__(self,operation,count=1):
        self.op = operation # should not return anything
        self.threads = [None for i in range(count)]
        self.running = [False]
        running_closure = self.running
        def theloop():
            while running_closure[0]:
                self.op()
        self.loop = theloop

    def start(self):
        if self.running[0] == True:
            raise NameError('waiter already started')
        print('(waiter)start')
        self.running[0] = True
        t = self.threads
        for i in range(len(t)):
            t[i] = th.Thread(target=self.loop)
            t[i].daemon = True # thread will exit when main thread exits
            t[i].start()

    def stop(self):
        print('(waiter)stop')
        self.running[0] = False
        t = self.threads
        for i in range(len(t)):
            t[i] = None

import os
def generate_and_save(count=10):
    directory = './generated/'

    # check if path exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_files = len(os.listdir(directory))

    if num_files < count:
        image, hc = imagine3()
        fname = directory + hc + '.jpg'

        cv2.imwrite(fname, image*255.,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print(fname,'saved...')

    else:
        time.sleep(.5)

def deadpool(count=10):
    # global wt
    # wt = Waiter(generate_and_save)
    # wt.start()

    while True:
        generate_and_save(count)
