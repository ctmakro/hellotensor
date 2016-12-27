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
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
import math
import keras.backend as K

batch_size = 64
nb_classes = 10
nb_epoch = 200
data_augmentation = True

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

model = Sequential()

model.add(Convolution2D(32, 5, 5,input_shape=(32,32,3))) #norma conv

# model.add(Convolution2D(64, 3, 3, border_mode='same',
#                         input_shape=X_train.shape[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 5, 5)) #norma conv
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 5, 5)) #norma conv
model.add(Activation('relu'))

# model.add(AtrousConvolution2D(128, 3, 3, atrous_rate=(4,4)))
# model.add(Activation('relu'))

model.add(Convolution2D(16, 1, 1)) #norma conv
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(10, 1, 1)) #norma conv
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.03, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train-=0.5
X_test-=0.5

# data_augmentation=False

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

Cb=keras.callbacks.Callback
class Mycb(Cb):
    def __init__(self):
        super(Cb, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
        lr=K.get_value(self.model.optimizer.lr)
        logs['lr'] = lr

#data_augmentation = False
def r(ep=10,bs=100,maxlr=0.05):
    batch_size=bs
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=ep,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
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
        import keras.callbacks as c
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=ep,
                            validation_data=(X_test, Y_test),
                            callbacks=[
                                gen_sgdr_callback(maxlr=maxlr), # adjust lr
                                Mycb(), # export lr
                                c.TensorBoard(log_dir='./logs2', histogram_freq=0, write_graph=False, write_images=False)
                            ],
                            )
