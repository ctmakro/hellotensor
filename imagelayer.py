from keras import backend as K
from keras.engine.topology import Layer
import keras
import keras.regularizers
import numpy as np
import math

class ImageLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ImageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=self.output_dim,
                                 initializer='uniform',
                                 trainable=True)
        super(ImageLayer, self).build(input_shape)

    def call(self, x, mask=None):
        xcount = K.shape(x)[0]
        return K.reshape(self.W,(xcount,)+self.output_dim) # return the weights directly

    def get_output_shape_for(self, input_shape):
        return (None,) + self.output_dim

class MultipleInputLayer(Layer):
    def __init__(self, output_dim,W_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.W_regularizer = keras.regularizers.get(W_regularizer)

        super(MultipleInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=self.output_dim,
                                 initializer='uniform',
                                 regularizer=self.W_regularizer,
                                 trainable=True)
        super(MultipleInputLayer, self).build(input_shape)

    #assume output dim is (3000,3)
    def call(self, x, mask=None):
        # xcount = K.shape(x)[0]
        return self.W
        # return K.reshape(self.W,(xcount,)+self.output_dim[-1])
        # return the weights directly

    def get_output_shape_for(self, input_shape):
        return self.output_dim

    def get_config(self):
            config = {
                      'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                      }
            base_config = super(MultipleInputLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

# input angle, output height on that angle. height map is described with fourier coefficients.
class SineLayer(Layer):
    def __init__(self, output_dim, max_freq=32, **kwargs):
        self.output_dim = output_dim
        self.coefs = max_freq
        super(SineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1]!=1:
            raise NameError('last dim of input shape is not 1. please provide one angle per datapoint')

        coefs = self.coefs
        # Create a trainable weight variable for this layer.
        self.Wsin = self.add_weight(shape=(coefs,self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.Wcos = self.add_weight(shape=(coefs,self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

        self.freq_multiplier = K.arange(coefs,dtype='float32')

        super(SineLayer, self).build(input_shape)

    def call(self, x, mask=None):
        angle = x
        coefs = self.coefs

        # assume coefs == 64
        #(assume angle shape is (128,1))
        # freq_m shape is (64,)
        # angle * freq_m(64,) = (128,64)

        # self.wsin = (64,10)
        coef1 = K.dot(K.sin(angle * self.freq_multiplier), self.Wsin)
        coef2 = K.dot(K.cos(angle * self.freq_multiplier), self.Wcos)

        # coef1 = (128,10)

        return coef1 + coef2

        # output 10 numbers

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-1]+(self.output_dim,)
