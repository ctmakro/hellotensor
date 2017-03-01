from __future__ import print_function

print('vgg16_playground.py loading...')

# from keras.optimizers import *
import keras
import math
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2

import canton as ct
# from canton import *

print('importing VGG19...')
from keras.applications.vgg16 import VGG16 as VGG19

starry_night = cv2.imread('starry_night.jpg').astype('float32') / 255. - .5
guangzhou = cv2.imread('DSC_0896cs_s.jpg').astype('float32') / 255. - .5

def feed_gen(output_size=[512,512]):
    # all the logic

    print('output size chosen:',output_size)

    # create white_noise_image
    global white_noise_image
    white_noise_image = tf.Variable(
        tf.random_normal([1]+output_size+[3], stddev=1e-12),
        dtype=tf.float32,name='white_noise_image')

    # the model to descent the white noise image
    global vggmodel_d
    vggmodel_d = VGG19(include_top=False, weights='imagenet', input_tensor=white_noise_image)
    vggmodel_d.summary()

    reference_image = ct.ph([None,None,3])
    # the model to extract style representations
    vggmodel_e = VGG19(include_top=False, weights='imagenet', input_tensor=reference_image)

    print('VGG models created.')

    def into_variable(value):
        v = tf.Variable(initial_value=value)
        sess = ct.get_session()
        sess.run([tf.variables_initializer([v])])
        return v

    def get_representations(vggmodel):

        # activations of each layer, 5 layers for style capture, 1 layer for content capture.
        layer_for_styles = list(filter(lambda x:'conv1' in x.name or 'block5_conv3' in x.name, vggmodel.layers))
        style_activations = [i.output for i in layer_for_styles]
        layer_for_content = ['block5_conv2']
        content_activations = [vggmodel.get_layer(l).output for l in layer_for_content]

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

        return gram_matrices,content_activations

    # get the gram matrices of the style reference image
    style_gram_matrices, content_activations = get_representations(vggmodel_e)

    # image shape manipulation: from HWC into NHWC
    sn = starry_night.view()
    sn.shape = (1,) + sn.shape

    sess = ct.get_session()
    sess.run(tf.global_variables_initializer())
    gram_ref = sess.run([style_gram_matrices],
        feed_dict={reference_image:sn})[0]

    print('reference style gram matrices calculated.')

    # load style references into memory
    style_references = [into_variable(gr) for gr in gram_ref]

    print('reference style gram matrices loaded into memory as variables.')

    # get content representation of the content image
    gz = guangzhou.view()
    gz.shape = (1,) + gz.shape

    reference_content_activations = sess.run([content_activations],
        feed_dict={reference_image:gz})[0]

    print('reference content representations calculated.')

    # load content reps into memory
    reference_content_activations = [into_variable(rca) for rca in reference_content_activations]
    print('reference content activations loaded into memory as variables.')

    # calculate losses of white_noise_image's style wrt style references:
    white_gram_matrices, white_content_activations = get_representations(vggmodel_d)

    def square_loss(g1,g2): # difference between two gram matrix, used as style loss
        return tf.reduce_sum((g1-g2)**2)

    white_style_losses = [square_loss(white_gram_matrices[idx],style_references[idx])
        for idx, gs in enumerate(style_references)]

    # calculate losses of white_noise_image's content wrt content reference:
    white_content_losses = [tf.reduce_mean((reference_content_activations[idx] - white_content_activations[idx])**2)
        for idx, _ in enumerate(reference_content_activations)]

    def amplitude_penalty(tensor, ceiling=0.499, floor=-0.499):
        p = tf.maximum(
            white_noise_image - ceiling,0) + tf.maximum(
            floor - white_noise_image,0)
        return tf.reduce_mean(p)

    def proportional_loss(lis): # similar to reduce mean, adds penalty if imbalance.
        mean_loss = tf.reduce_mean(lis)
        pro_loss = tf.reduce_mean([(l - mean_loss)**2 for l in lis])
        return mean_loss + pro_loss * 5

    white_amplitude_penalty = amplitude_penalty(white_noise_image)

    white_loss = proportional_loss([proportional_loss(white_style_losses) , tf.reduce_mean(white_content_losses) * .004])

    white_loss += tf.reduce_mean(white_amplitude_penalty**2)*10000

    # minimize loss by gradient descent on white_noise_image
    learning_rate = tf.Variable(0.01)

    adam = tf.train.AdamOptimizer(learning_rate)
    #adam = tf.train.MomentumOptimizer(learning_rate,momentum=0.95)
    print('connecting momentum sgd optimizer...')
    descent_step = adam.minimize(white_loss,var_list=[white_noise_image])

    # initialize the white_noise_image
    # sess.run([tf.variables_initializer([white_noise_image])])
    sess.run(tf.global_variables_initializer())

    print('white_noise_image initialized.')

    def feed(lr=.01):
        nonlocal white_loss,descent_step,learning_rate
        sess = ct.get_session()
        res = sess.run([descent_step,white_loss],
        feed_dict={learning_rate:lr})
        loss = res[1]
        return loss

    print('feed function generated.')
    return feed

feed = feed_gen(output_size=list(guangzhou.shape[0:2]))

print('Ready to dream.')

def r(ep=10,maxlr=.01):
    import time
    t = time.time()
    for i in range(ep):
        t = time.time()
        print('ep',i)
        lr = maxlr #* (math.cos((i%10)*math.pi/9)+1)/2 + 1e-9
        print('lr:',lr)
        loss = feed(lr=lr)
        t = time.time()-t
        print('loss: {:6.6f}, {:6.4f}/run, {:6.4f}/s'.format(loss,t,1/t))

        if i%5==0:
            saveit = True if i%50==0 else False #every 100 ep
            show(save=saveit)

show_counter = 0
show_prefix = str(np.random.choice(1000))

def show(save=False):
    sess = ct.get_session()
    res = sess.run(vggmodel_d.input)
    image = res[0]
    image+=0.5
    cv2.imshow('result',image)
    cv2.waitKey(1)
    cv2.waitKey(1)
    if save:
        global show_counter,show_prefix
        cv2.imwrite('./log/'+show_prefix+'_'+str(show_counter)+'.jpg',image*255.)
        show_counter+=1
    return image

def replace_original():
    sess = ct.get_session()
    rg = guangzhou.view()
    rg.shape = (1,) + guangzhou.shape
    v = tf.Variable(rg)
    sess.run(tf.variables_initializer([v]))
    sess.run(tf.assign(white_noise_image,v))

def clear():
    ct.get_session().run(tf.variables_initializer([white_noise_image]))
    print('white noise image cleared.')
