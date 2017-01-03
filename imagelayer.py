from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

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
        return K.reshape(self.W,(1,)+self.output_dim) # return the weights directly

    def get_output_shape_for(self, input_shape):
        return (1,) + self.output_dim
