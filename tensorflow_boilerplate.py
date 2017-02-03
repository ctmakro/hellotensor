
import tensorflow as tf
import numpy as np

def make_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def make_bias(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(nip,nop,dim,std=1):
    def conv(i):
        W = make_weight([dim,dim,nip,nop])
        b = make_bias([nop])
        c = tf.nn.conv2d(i, W, strides=[1, std, std, 1], padding='SAME')
        return c + b
    return conv

def relu(i):
    return tf.nn.relu(i)

def resconv(i,nip,nop,std=1):
    nop4 = int(nop/4)
    inp = i

    i = relu(i)
    i = conv2d(nip,nop4,1,std=std)(i)
    i = relu(i)
    i = conv2d(nop4,nop4,3)(i)
    i = relu(i)
    i = conv2d(nop4,nop,1)(i)

    if nip==nop and std==1:
        inp = inp
    else:
        inp = conv2d(nip,nop,1,std=std)(inp)

    return inp+i

def dense(nip,nop):
    def dens(i):
        W = make_weight([nip,nop])
        b = make_bias([nop])
        d = tf.matmul(i,W)+b
        return d
    return dens

def avgpool2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def batch_accuracy(pred,gt):
    correct_vector = tf.equal(
    tf.argmax(pred,1), tf.argmax(gt,1)
    )
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def summary(varlist=None):
    if varlist is None:
        varlist = tf.trainable_variables()

    varcount = 0
    for v in varlist:
        shape = v.get_shape()
        varcount += shape.num_elements()
    print(varcount,'trainable variables')

def Adam(lr=1e-3):
    return tf.train.AdamOptimizer(lr)

def TrainWith(loss,optimizer):
    gradients = optimizer.compute_gradients(loss)
    update = optimizer.apply_gradients(gradients)
    # descend all variables wrt their gradient to reduce loss
    return update

class StatusBar(object):
    def __init__(self,length):
        import progressbar as p2

        bar = p2.ProgressBar(max_value=length,widgets=[
        p2.Percentage(),
        ' ',
        p2.SimpleProgress(format='%(value)s/%(max_value)s'),
        p2.Bar('>'),
        p2.Timer(),
        ' ',
        p2.ETA(),
        ' ',
        p2.DynamicMessage('loss'),
        ' ',
        p2.DynamicMessage('acc')
        ])

        self.bar = bar
        self.length = length

    def update(self,mile,**kwargs):
        self.bar.update(mile,**kwargs)

import time
class TrainTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def stamp(self):
        return time.time() - self.start_time

    def epstart(self):
        self.epstart_time = self.stamp()

    def epend(self):
        epduration = self.stamp() - self.epstart_time
        return epduration

    def epend_report(self,samples):
        length = samples
        eptime = self.epend()
        now = self.stamp()
        print('{} sample in {:6.2f}s, {:6.4f}/sample, {:6.2f}/s - {:6.2f}s total'.format(
        length,eptime,eptime/length,length/eptime,now))
