
import tensorflow as tf
from tensorflow_session import get_session
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

def categorical_cross_entropy(y,gt):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, gt))
    return loss

def make_function(i_tensors,o_tensors):
    def f(values):
        sess = get_session()
        return sess.run(o_tensors,feed_dict={i_tensors:values})
    return f

class StatusBar:
    def __init__(self,length):
        import progressbar as p2

        class modifiedDynamicMessage(p2.DynamicMessage):
            def __call__(self, progress, data):
                string = data['dynamic_messages'][self.name]
                if string is not None:
                    return string
                else:
                    return ''
                    return self.name + ': ' + 6 * '-'

        bar = p2.ProgressBar(max_value=length,widgets=[
        p2.Percentage(),
        ' ',
        p2.SimpleProgress(format='%(value)s/%(max_value)s'),
        p2.Bar('>'),
        p2.Timer(format='%(elapsed)s'),
        ' ',
        p2.ETA(
            format='ETA %(eta)s',
            format_not_started='ETA --:--:--',
            format_finished='TIM %(elapsed)s'),
        ' ',
        modifiedDynamicMessage('msg')
        ])

        self.bar = bar
        self.length = length

    def update(self,mile,**kwargs):
        self.bar.update(mile,**kwargs)

    def finish(self):
        self.bar.finish()

import time
class TrainTimer:
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
        print('{:>10} sample in {:6.2f}s, {:6.4f}/sample, {:6.2f}/s - {:6.2f}s total'.format(
        length,eptime,eptime/length,length/eptime,now))

class ModelRunner:
    def __init__(self,inputs,outputs,gt=None,sess=None):
        self.inputs = inputs
        self.outputs = outputs

        self.acc = None
        self.loss = None
        self.sess = sess # default none
        self.gt = gt

        if self.sess is None:
            self.sess = get_session()

        self.train_step = None
        self.reporter = None

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def set_loss(self,loss_function):
        self.loss = loss_function

    def set_acc(self,acc_function):
        self.acc = acc_function

    def set_training_operation(self,op=None):
        if op is None:
            # default training method
            def default_training_operation():
                if self.loss is None or self.optimizer is None:
                    raise NameError('please specify loss and optimizer, or provide a training op.')

                loss,optimizer = self.loss,self.optimizer
                gradients = optimizer.compute_gradients(loss)
                update = optimizer.apply_gradients(gradients)
                # descend all variables wrt their gradient to reduce loss
                return update

            op = default_training_operation()

        self.train_step = op

    def ready_to_train(self):
        if self.train_step is None: #if train_step not provided
            self.set_training_operation() # default training operation

        self.sess.run(tf.global_variables_initializer())

    def one_step(self,inputs,gt=None,train=False):
        # train the model on one datum/minibatch,
        # evaluate metrics, update parameters
        train_step = self.train_step
        loss = self.loss
        acc = self.acc

        want_to_execute = [loss]

        if train == True:
            want_to_execute = [train_step] + want_to_execute
        if self.acc is not None:
            want_to_execute += [acc]

        feed_dict = {self.inputs:inputs}

        if self.gt is not None and gt is not None:
            # feed gt only if we need it.
            feed_dict[self.gt] = gt

        if self.gt is None != gt is None:
            # if specified not provided, or provided not specified
            raise NameError('something wrong with gt or self.gt, check your code.')

        result = self.sess.run(want_to_execute,
        feed_dict=feed_dict)

        if train == True:
            result = result[1:]

        return result

    def set_step_result_reporter(self,reporter):
        self.reporter = reporter

    def step_result_report(self,result):
        if self.reporter is None:
            def reporter(result):# default reporter
                # expected input: result -> [loss,acc] or [loss]
                res = result

                loss=res[0]
                acc =res[1] if len(res)>1 else None

                message = 'loss:{:6.4f}'.format(loss)
                if acc is not None: message += ' acc:{:6.4f}'.format(acc)
                return message
            self.set_step_result_reporter(reporter)

        return self.reporter(result)

    def defaultEpochRunner(self,xtrain,ytrain=None,xtest=None,ytest=None):
        def r(ep=100,bs=128):
            length = len(xtrain) # length of one epoch run

            if ytrain is None:
                print('No labels/gt provided.')

            if ytrain is not None and length!=len(ytrain):
                # if shit just happened
                print('len(xtrain) != len(ytrain), check your code.')
                return

            batch_size = bs
            timer = TrainTimer()

            for i in range(ep):
                timer.epstart()
                print('--------------\nEpoch',i)

                bar = StatusBar(length)

                for j in range(0, length, batch_size):
                    # train one minibatch
                    inputs = xtrain[j:j+batch_size]
                    labels = ytrain[j:j+batch_size] if ytrain is not None else None

                    res = self.one_step(inputs,labels,train=True)
                    message = self.step_result_report(res)

                    bar.update(min(j+batch_size,length),
                    msg = message
                    )

                bar.finish()
                timer.epend_report(length)

                if xtest is not None and ytest is not None:
                    # test set
                    timer.epstart()
                    res = self.one_step(xtest,ytest,train=False)
                    timer.epend_report(len(xtest))
                    message = self.step_result_report(res)
                    print(message)
        return r
