from __future__ import print_function

from tensorflow_boilerplate import *
# from tensorflow_session import get_session
import tensorflow as tf
import numpy as np

def build_model():
    inp = tf.placeholder(tf.float32, shape=[None, 784])

    i = inp
    i = tf.reshape(i,[-1,28,28,1]) # reshape into 4d tensor

    i = conv2d(1,8,3)(i)

    i = resconv(i,8,8)
    i = resconv(i,8,16,std=2)

    i = resconv(i,16,16)
    i = resconv(i,16,32,std=2)

    i = resconv(i,32,32)
    i = resconv(i,32,64,std=2)

    i = resconv(i,64,10)

    i = tf.reduce_mean(i,[1,2]) # 2d tensor (N, onehot)

    out = i
    return inp,out

x,y = build_model()
gt = tf.placeholder(tf.float32, shape=[None, 10])

mr = ModelRunner(inputs=x,outputs=y,gt=gt)
mr.set_loss(categorical_cross_entropy(y,gt)) # loss(y,gt)
mr.set_optimizer(Adam(1e-3))
mr.set_acc(batch_accuracy(y,gt)) # optional. acc(y,gt)
mr.ready_to_train()

def mnist_data():
    print('loading mnist...')
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

    print('mnist loaded.')
    return X_train,Y_train,X_test,Y_test

xtrain,ytrain,xtest,ytest = mnist_data()

summary()

def get_graph_variables(op):
    var_list = op.graph.get_collection('trainable_variables')
    return var_list

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6}'.format(shape_lists[i],num_elements[i]))

    print('{:>25}  ->  {:<6}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables)+' free variables'))

r = mr.defaultEpochRunner(xtrain,ytrain,xtest,ytest)
# r = mr.defaultEpochRunner(xtrain,ytrain)
