import tensorflow as tf
import canton as ct
import numpy as np

input_variable = tf.Variable(np.random.normal(loc=0,scale=1,size=[1,256,256,3]
    ).astype('float32'))

conv = ct.Conv2D(3,16,3)
shared_conv = ct.Conv2D(16,16,3)

i = conv(input_variable)
i = shared_conv(i)
i = shared_conv(i)

class DoubleConv(ct.Can):
    def __init__(self):
        super().__init__() # init base class
        self.convs = [ct.Conv2D(3,16,3),ct.Conv2D(16,3,3)] # define conv2d cans
        self.incan(self.convs) # add as subcans
    def __call__(self,i):
        i = self.convs[0](i)
        i = self.convs[1](i)
        return i

input_variable2 = tf.Variable(np.random.normal(loc=0,scale=1,size=[1,256,256,3]
    ).astype('float32'))

dc = DoubleConv()
i2 = dc(input_variable2)
i2 = dc(i2)
