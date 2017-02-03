from __future__ import print_function

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

def make_weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def make_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(nip,nop,dim):
    def conv(i):
        W = make_weight([dim,dim,nip,nop])
        b = make_bias([nop])
        c = tf.nn.conv2d(i, W, strides=[1, 1, 1, 1], padding='SAME')
        return c + b
    return conv

def dense(nip,nop):
    def dens(i):
        W = make_weight([nip,nop])
        b = make_bias([nop])
        d = tf.matmul(i,W)+b
        return d
    return dens

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
x2d = tf.reshape(x,[-1,28,28,1]) # reshape incoming into 2d

h_conv1 = tf.nn.relu(conv2d(1,16,5)(x2d))
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(16,32,5)(h_pool1))
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flatten = tf.reshape(h_pool2,[-1,7*7*32])

h_fc1 = tf.nn.relu(dense(7*7*32,64)(h_pool2_flatten))
h_fc2 = dense(64,10)(h_fc1)

# prediction = tf.matmul(x,W) + b # linear layer
prediction = h_fc2

gt = tf.placeholder(tf.float32, shape=[None, 10])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
prediction, gt))
# loss function: softmax + cross_entropy

correct_prediction = tf.equal(
tf.argmax(prediction,1), tf.argmax(gt,1)
)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss) # modify all variables wrt their gradient to reduce loss

sess.run(tf.global_variables_initializer())

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

# reshape into 2d
X_train = X_train.reshape(X_train.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

def categorical(tensor,cat):
    newshape = tuple(tensor.shape[0:1])+(cat,)
    print(newshape)
    new = np.zeros(newshape,dtype='float32')
    for i in range(cat):
        new[:,i] = tensor[:] == i
    return new

Y_train = categorical(y_train,10)
Y_test = categorical(y_test,10)

print('Y_train shape:',Y_train.shape)
print(Y_train[0])

def r(ep=100,bs=128):
    import time
    import progressbar as p2

    start_time = time.time()
    def stamp():
        return time.time()-start_time

    length = len(X_train) # length of one epoch run
    batch_size = bs

    for i in range(ep):
        epstart_time=stamp()
        print('--------------')
        print('Epoch',i)

        bar = p2.ProgressBar(max_value=length,widgets=[
        p2.Percentage(),
        ' ',
        p2.SimpleProgress(format='%(value)s/%(max_value)s'),
        p2.Bar('$'),
        p2.Timer(),
        ' ',
        p2.ETA(),
        ' ',
        p2.DynamicMessage('loss'),
        ' ',
        p2.DynamicMessage('acc')
        ])
        # bar.update(0)

        for j in range(0, length, batch_size):
            # train one minibatch
            inputs = X_train[j:j+batch_size]
            labels = Y_train[j:j+batch_size]
            res = sess.run([accuracy,loss,train_step],feed_dict={x:inputs, gt:labels})
            # train_step.run(feed_dict={x:inputs, gt:labels})

            bar.update(min(j+batch_size,length),
            loss=res[1],
            acc =res[0]
            )

        print('')

        #report time
        now = stamp()
        eptime = now - epstart_time
        print('{} sample in {:6.2f}s, {:6.4f}/sample, {:6.2f}/s - {:6.2f}s total'.format(
        length,eptime,eptime/length,length/eptime,now))

        # test set
        res = sess.run([accuracy,loss],feed_dict={x:X_test, gt:Y_test})

        print('test loss:{:6.4f} acc:{:6.4f}'.format(res[1],res[0]))
