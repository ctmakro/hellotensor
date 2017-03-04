'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''
from __future__ import print_function

# tf server part

# import tensorflow as tf
#
# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)
#
# from keras import backend as K
# K.set_session(sess)

# end tf server part

import numpy as np

import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 500
#nb_classes = 10
nb_classes = 2
nb_epoch = 1
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3


from load_drone_data import load_data
ximages,yvalues = load_data(False)

# now split
# 5/6 train, 1/6 test

total = ximages.shape[0]
split = int(total*45/50)

X_train = ximages[0:split]
y_train = yvalues[0:split]

X_test = ximages[split:total]
y_test = yvalues[split:total]

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print('y_train shape:',y_train.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#exit()

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train = np.reshape(y_train,y_train.shape+(1,))
Y_test = np.reshape(y_test,y_test.shape+(1,))
# Y_train = y_train
# Y_test = y_test
print('Y_train after conversion:',Y_train.shape)



model = Sequential()

model.add(Convolution2D(16, 7, 7, #border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Convolution2D(16, 7, 7))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))

# this should cover 32x32 area

model.add(Convolution2D(8, 1, 1))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(1, 1, 1))
model.add(Activation('relu'))

# model.add(Dropout(0.25))
# generate 2 act. map
nb_classes = 2

# model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))

#model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# # why drop out..
#
# # end
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

def my_fancy_loss(y_true, y_pred):
    # ?, 3, 3, 1
    # y_true should be between 0,1
    # y_pred should be 0,+inf
    yt = tf.reshape(y_true,[-1,9])
    yp = K.softmax(tf.reshape(y_pred,[-1,9]))

    return - K.mean(K.log(yp+1e-8)*yt,axis=-1)
    #return K.mean(K.square(y_pred - y_true), axis=-1)

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(#loss='categorical_crossentropy',
                loss=my_fancy_loss,
                optimizer=sgd,
                # metrics=['accuracy'],
                # metrics=['']
                )



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_train -= 0.5
X_test /= 255
X_test -= 0.5

# load pretrained
# model = load_model('evenbetter.h5')
model.summary()

#exit()

def c1w_show():
    import vis
    c1w = model.layers[0].get_weights()[0]
    vis.weightshow(c1w)

def getimage(index):
    xi = ximages[index]
    i = np.reshape(xi,(xi.shape[0],xi.shape[1]))
    return i

def imshow(index):
    import matplotlib.pyplot as plt
    plt.imshow(getimage(index),cmap='gray')
    plt.show(block=False)

def imrange(start=0,count=2):
    import matplotlib.pyplot as plt
    import math
    s = count+5
    fig0, plots = plt.subplots(count,s,subplot_kw={'xticks': [], 'yticks': []})
    fig0.subplots_adjust(hspace=0.1, wspace=0.01)
    index=0
    for line in plots:
        for i in line:
            picnum = start+index
            p = predict(picnum)
            p = 1 if p[0,1]>p[0,0] else 0
            real = yvalues[picnum,0]
            if p==real:
                #nice guess
                i.imshow(getimage(picnum),cmap='gray')
            else:
                i.imshow(getimage(picnum),cmap='hot')
            i.set_title(str(picnum)+': '+str(p)+'('+str(real),fontsize=5)
            print(p,real)
            index+=1
    plt.show(block=False)

def predict(index):
    xi = ximages[index]
    i = np.reshape(xi,(1,xi.shape[0],xi.shape[1],1))
    i = i.astype('float32')
    i/=255
    i-=0.5
    return model.predict(i)

def ps():
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')

def e():
    exit()

import math

def gen_sgdr_callback(minlr=0.0001,maxlr=0.05,t0=5,tm=2): # https://arxiv.org/pdf/1608.03983.pdf
    print('generating SGDR:',minlr,maxlr,t0,tm)
    def lr(epoch):
        if epoch > lr.lastepoch: # update tcur only if necessary
            lr.lastepoch=epoch
            lr.tcur+=1
        if epoch < lr.lastepoch:# what?
            lr.tcur=0
        if lr.tcur > lr.tz:
            lr.tcur=0
            lr.tz = int(lr.tz * tm)
        newlr= minlr+0.5*(maxlr-minlr)*(1+math.cos(float(lr.tcur)/lr.tz*math.pi))
        print('lr:',newlr,'@ep',epoch,'phase:',float(lr.tcur)/lr.tz)
        return newlr
    lr.tz=t0
    lr.tcur=0
    lr.lastepoch=0 # python does not have real closure. sucks!
    return keras.callbacks.LearningRateScheduler(lr)

import keras.callbacks as c
Cb=c.Callback
class Mycb(Cb):
    def __init__(self):
        super(Cb, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
        lr=K.get_value(self.model.optimizer.lr)
        logs['lr'] = lr

data_augmentation = False

def gen_callback_stack(maxlr,t0=5):
    return [
        gen_sgdr_callback(maxlr=maxlr,t0=t0), # adjust lr
        Mycb(), # export lr
        c.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=True)
    ]

def r(ep=10,bs=100,maxlr=0.05,t0=5):
    batch_size=bs
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=ep,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=gen_callback_stack(maxlr=maxlr)
                  )
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=ep,
                            validation_data=(X_test, Y_test),
                            callbacks=gen_callback_stack(maxlr=maxlr,t0=t0)
                            )
