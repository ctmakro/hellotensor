from __future__ import print_function

print('vgg16_playground.py loading...')

from keras.models import *
from keras.layers import *
from keras.layers import *

# from keras.optimizers import *
import keras
import math
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2

print('importing VGG16...')
from keras.applications.vgg16 import VGG16

starry_night = cv2.imread('starry_night.jpg').astype('float32') / 255. - .5

def feed_gen(output_size=[512,512]):
    # all the logic
	
	print('output size chosen:',output_size)

    # create white_noise_image
    global white_noise_image
    white_noise_image = tf.Variable(
        tf.random_normal([1]+output_size+[3], stddev=0.1),
        dtype=tf.float32,name='white_noise_image')

    # the model to descent the white noise image
    global vggmodel_d
    vggmodel_d = VGG16(include_top=False, weights='imagenet', input_tensor=white_noise_image)
    vggmodel_d.summary()

    style_reference_image = Input((None,None,3))
    # the model to extract style representations
    vggmodel_e = VGG16(include_top=False, weights='imagenet', input_tensor=style_reference_image)

    print('VGG models created.')

    def into_variable(value):
        v = tf.Variable(initial_value=value)
        sess = K.get_session()
        sess.run([tf.variables_initializer([v])])
        return v

    def get_representations(vggmodel):

        # activations of each layer, 5 layers for style capture, 1 layer for content capture.
        style_activations = [vggmodel.get_layer('block'+str(i+1)+'_conv1').output for i in range(4)] # block_1..5_conv1
        content_activation = vggmodel.get_layer('block4_conv2').output

        def gram_4d(i):
            # calculate gram matrix (inner product) of feature maps.
            # where gram[n1, n2] is the correlation between two out of n features.

            # for example two feature map are each sensitive to tree and flower,
            # then gram[tree, flower] tells you how tree and flower are
            # correlated in the layer activations.
            # in other words, how likely tree and flower will appear together.

            # this correlation does not depend on position in image,
            # and that's why we can calculate style loss globally.
            # in other words, we don't care about the exact position of features,
            # but how likely each of them appear with another.

            # assume input is 4d tensor of shape [1, h, w, f]
            s = tf.shape(i)

            # reshape into feature matrix of shape [h*w, f]
            fm = tf.reshape(i,[s[1]*s[2],s[3]])

            # inner product
            gram = tf.matmul(tf.transpose(fm),fm) # [f, f]

            # because h*w*f elements are included in computing the inner product,
            # we have to normalize the result:
            gram = gram / tf.cast((s[1]*s[2]*s[3])*2,tf.float32)
            return gram

        gram_matrices = [gram_4d(i) for i in style_activations]

        return gram_matrices

    # get the gram matrices for the style reference image
    style_gram_matrices = get_representations(vggmodel_e)

    # image shape manipulation: from HWC into NHWC
    sn = starry_night.view()
    sn.shape = (1,) + sn.shape

    sess = K.get_session()
    gram_ref = sess.run([style_gram_matrices],
        feed_dict={vggmodel_e.get_input_at(0):sn})[0]

    print('reference style gram matrices calculated.')

    # load style references into memory
    style_references = [into_variable(gr) for gr in gram_ref]

    print('reference style gram matrices loaded into memory as variables.')

    # calculate losses of white_noise_image's style wrt style references:
    white_gram_matrices = get_representations(vggmodel_d)

    def square_loss(g1,g2): # difference between two gram matrix, used as style loss
        return tf.reduce_sum((g1-g2)**2)

    white_style_losses = [square_loss(white_gram_matrices[idx],style_references[idx])
        for idx, gs in enumerate(style_references)]

    white_loss = tf.reduce_mean(white_style_losses)

    # minimize loss by gradient descent on white_noise_image
    lr = 0.1
    adam = tf.train.AdamOptimizer(lr)
    print('Adam Optimizer lr = {}'.format(lr))
    descent_step = adam.minimize(white_loss,var_list=[white_noise_image])

    # initialize the white_noise_image
    sess.run([tf.variables_initializer([white_noise_image])])

    print('white_noise_image initialized.')

    def feed():
        nonlocal white_loss,descent_step
        sess = K.get_session()
        res = sess.run([descent_step,white_loss])
        loss = res[1]
        return loss

    print('feed function generated.')
    return feed

feed = feed_gen(output_size=[768,768])

print('Ready to dream.')

def r(ep=10):
    for i in range(ep):
        print('ep',i)
        loss = feed()
        print('loss:',loss)

        if i%2==0:
            show()

def show():
    sess = K.get_session()
    res = sess.run(vggmodel_d.input)
    image = res[0]
    image+=0.5
    cv2.imshow('result',image)
    cv2.waitKey(2)
