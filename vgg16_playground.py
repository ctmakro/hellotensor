from __future__ import print_function

print('vgg16_playground.py loading...')

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D,AveragePooling2D

from keras.optimizers import SGD,Adam,Adadelta,Nadam
from keras.utils import np_utils
import keras
import math
import keras.backend as K
import sgdr
import numpy as np
from keras.models import load_model
import cv2

# lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)

print('importing VGG16...')
from keras.applications.vgg16 import VGG16
vggmodel = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None,None,3))

print('imported.')
def pooling_input_to(m):
    # freeze all layers
    for l in m.layers:
        l.trainable = False

    # shortcut the vgg model
    m = Model(input=m.input,output=m.get_layer('block3_conv2').output)

    some_input = Input(shape=(None,None,None,))
    pooled = AveragePooling2D(pool_size=(4, 4),strides=(2,2),border_mode='same')(some_input) # the picture layer
    activations = m(pooled)
    return Model(input=some_input,output=activations)

model = pooling_input_to(vggmodel)

def neural_difference(img1,img2):
    # assume 0.0-1.0 RGB

    dsf = 8 # down-sampling factor
    if img1.shape[0]<dsf or img1.shape[1]<dsf or img2.shape[0]<dsf or img2.shape[1]<dsf:
        print('area smaller than 8x8')
        return False

    i1 = img1-0.5
    i2 = img2-0.5
    [act1,act2] = model.predict(np.array([i1,i2]))
    print('nd_activation shape:',act1.shape,act2.shape)

    # if act1.shape[]

    diff = (act1-act2)
    diff = np.mean(diff*diff)
    # print('diff:',diff)
    return diff

def rebuild():
    model = Sequential()

    model.add(Convolution2D(16, 5, 5,
                            input_shape=(None,None,3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # this should cover 32x32 area

    # model.add(Convolution2D(64, 5, 5))
    # model.add(Activation('relu'))
    #
    # model.add(Convolution2D(16, 1, 1))
    # model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    # model.add(Convolution2D(10, 1, 1))
    # model.add(Flatten())

    trained.summary()
    model.summary()

    model.set_weights(trained.get_weights())
    print('weight sucessfully set.')
    return model


# trained = load_model('small_cifar.h5')
# model = rebuild()

model.summary()

def flatten_multidim_arr_into_image(arr):
    import cv2
    uh,uw,depth = arr.shape

    patches = int(depth+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 1),dtype='float32')

    index = 0
    for row in range(height):
        if index>=depth-1:
            break
        for col in range(width):
            channels = arr[:,:,index:index+1]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,0:0+channels.shape[2]] = channels
            index+=1

    limit = 512
    if img.shape[0]>limit or img.shape[0]<limit/2:
        if img.shape[0]<limit/2:
            limit=int(limit/2)
        scale = img.shape[0]/limit
        img = cv2.resize(img,dsize=(int(img.shape[1]/scale),limit),interpolation=cv2.INTER_NEAREST)

    return img

def show_activations():
    pr = model.predict(flowers)[0]
    w = nm.layers[1].get_weights()[0]
    pr2 = model.predict(w.reshape((1,)+w.shape))[0]

    # print(pr.shape,pr2.shape)

    # c1 = model.layers[0]
    # w = c1.get_weights()[0]
    #
    # w = w.reshape((5,5,3*16))
    # imgw = flatten_multidim_arr_into_image(w)

    img = flatten_multidim_arr_into_image(pr)
    img2 = flatten_multidim_arr_into_image(pr2)

    img,img2 = img/np.max(img), img2/np.max(img2)

    cv2.imshow('img',img)
    cv2.imshow('img2',img2)
    cv2.waitKey(1)
    cv2.waitKey(1)

# flower = cv2.imread('../opencv_playground/flower.jpg').astype('float32')/255.
flower = cv2.imread('../opencv_playground/DSC_0896_512.jpg').astype('float32')/255.

# flower = cv2.resize(flower,dsize=(256,256))
flowers = flower.reshape((1,)+flower.shape) - 0.5

from imagelayer import ImageLayer
print('test predicting...')
target = model.predict(flowers)
print('input shape:',flowers.shape,'target shape:',target.shape)

null_input = Input(shape=(None,))
imagelayer = ImageLayer(output_dim=flower.shape)(null_input) # the picture layer
activations = model(imagelayer)

nm = Model(input=null_input,output=activations)
nm.summary()

# nm.layers[1].set_weights(weights=flowers)

import time
stamp = time.time()
def show():
    if time.time()-0.3 > stamp:
        global stamp
        stamp = time.time()
        imw = nm.layers[1].get_weights()[0]
        imw = cv2.resize(imw+.5,dsize=(512,512))
        cv2.imshow('im',imw)
        cv2.waitKey(1)

        show_activations()

def my_fancy_loss(y_true, y_pred):
    # ?, 3, 3, 1
    # y_true should be between 0,1
    # y_pred should be 0,+inf
    yt = y_true
    yp = y_pred

    return K.mean(K.square(yp - yt))

    # return K.mean(K.square(y_pred - y_true), axis=-1)

def r(ep=10,opt=None,lr=0.01,loss='mse'):
    from histlogger import EpochEndCallback
    callbacks=[EpochEndCallback(show)]

    sgd = SGD(lr=lr, decay=1e-4, momentum=0.95, nesterov=True)
    # opt = Adam()
    if opt is None:
        opt = sgd
        callbacks.append(sgdr.gen_scheduler(maxlr=lr,t0=10,tm=1))

    nm.compile(#loss='categorical_crossentropy',
                    loss=loss,
                    optimizer=opt,
                    # metrics=['accuracy'],
                    # metrics=['']
                    )

    nm.fit(np.array([0.0]),target,
              batch_size=1,
              nb_epoch=ep,
              callbacks=callbacks)
