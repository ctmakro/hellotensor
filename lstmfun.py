
from __future__ import print_function
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import random
import sys

model = 0

def get_seq(count=8192,base=2,length=10):
    print('generating sequences: {} examples, each sequence of length {}, with {}-based digits'.format(count,length,base))

    seqin = np.random.choice(base,(count,length,2))
    #(count,length_seq,2)
    seqout = seqin[:,:,0].copy()
    #(count,length_seq,1)

    carry = 0
    for i in range(length):
        # calc intermediate result
        imd = seqin[:,i,0] * seqin[:,i,1] + carry

        carry = np.floor(imd/base)
        seqout[:,i] = imd - carry * base

    sample_last_seqout = seqout[:,-1].copy()

    return seqin,sample_last_seqout

sequence_length = 21
number_base = 10
seqin,seqout = get_seq(count=4096,base=number_base,length=sequence_length)

# print(seqin,seqout)

print('seqin and seqout shape:',seqin.shape,seqout.shape)

print('now convert them to one hot..')
def one_hot(tensor,classes):
    heat = np.zeros(tensor.shape+(classes,))
    for i in range(classes):
        heat[...,i] = tensor[...] == i
    return heat

seqin = one_hot(tensor=seqin,classes=10)
seqout = one_hot(tensor=seqout,classes=10)

# print(seqin,seqout)
print('seqin, seqout:',seqin.shape,seqout.shape)

print('now reshaping seqin...')
def reshape_seqin(seqin):
    mod_seqin = np.reshape(seqin,seqin.shape[0:2]+((seqin.shape[2]*seqin.shape[3]),))
    return mod_seqin

mod_seqin = reshape_seqin(seqin)

print('seqin, reshaped_seqin, seqout:',seqin.shape,mod_seqin.shape,seqout.shape)
print(seqin[0],seqout[0])

def buildmodel():
    # build the model: a single LSTM
    print('Build model...')
    inp = Input(shape=(sequence_length,number_base*2,))
    i = inp
    i = LSTM(16)(i)
    lstmout = i

    i = Dense(32)(i)
    i = Activation('relu')(i)
    i = merge([i,lstmout],mode='concat')

    i = Dense(32)(i)
    i = Activation('relu')(i)
    i = merge([i,lstmout],mode='concat')

    i = Dense(number_base)(i)

    i = Activation('softmax')(i)

    model = Model(input=inp,output=i)

    model.summary()

    return model

model = buildmodel()

def loadmodel():
    print('loading model from file')
    global model
    model = load_model('naivelstm.h5')
    print('model loaded')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds += 1e-8
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

from histlogger import LoggerCallback

lcb = LoggerCallback(keys=[
{'loss':[]},
{'acc':[]}
]
)

def r(ep=1,opt=None,lr=0.05):
    if opt is None:
        opt = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=opt)
    model.fit(mod_seqin,seqout,batch_size=256,nb_epoch=ep,
    callbacks=[lcb]
    )

def test(i):
    print(model.predict(mod_seqin[i:i+1]),seqout[i])
#--------------------------------------------------------------------------------
# train the model, output generated text after each iteration
def re(epochs=1):
    for iteration in range(1, epochs+1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.5, 1.0]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

# loadmodel()
def generate(prefix='',length=100,diversity=1.0):
    print()
    print('----- diversity:', diversity)

    if prefix=='':
        start_index = random.randint(0, len(text) - maxlen - 1)
        prefix = text[start_index: start_index + maxlen]


    print('----- Generating with prefix: "' + prefix + '"','length',len(prefix))
    sys.stdout.write(prefix)

    context = prefix

    for i in range(length):
        x = np.zeros((1, maxlen, len(chars)))
        for idx, char in enumerate(context):
            x[0, idx, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]

        next_char_index = sample(preds, diversity)
        next_char = indices_char[next_char_index]

        context = context[1:]+next_char
        # context = prefix[-40:] # remove from left, to keep a length of 40
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
