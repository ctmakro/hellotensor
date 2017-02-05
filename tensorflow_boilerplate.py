
import tensorflow as tf
from tensorflow_session import get_session
import numpy as np
import time

def make_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name='W')

def make_bias(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial,name='b')

def conv2d(nip,nop,dim,std=1,usebias=True):
    def conv(i):
        W = make_weight([dim,dim,nip,nop])
        c = tf.nn.conv2d(i, W, strides=[1, std, std, 1], padding='SAME')
        if usebias == True:
            b = make_bias([nop])
            return c + b
        else:
            return c
    return conv

def relu(i):
    # return tf.nn.relu6(i)
    return tf.nn.relu(i)

# training state/mode mgmt -----------------------------------------------------
# borrowed from tflearn

def init_training_mode():
    # 'is_training' collection stores the training mode variable
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        tr_var = tf.Variable(False,name='is_training',trainable=False)
        tf.add_to_collection('is_training', tr_var)

        # 'is_training_ops' stores the ops to update training mode variable
        a = tf.assign(tr_var, True)
        b = tf.assign(tr_var, False)
        tf.add_to_collection('is_training_ops', a)
        tf.add_to_collection('is_training_ops', b)

def get_training_mode():
    init_training_mode() # create training mode variable, if not created
    coll = tf.get_collection('is_training')
    return coll[0]

def set_training_mode(mode):
    sess = get_session()
    init_training_mode()
    if mode == True:
        sess.run(tf.get_collection('is_training_ops')[0]) # 'a'
    else:
        sess.run(tf.get_collection('is_training_ops')[1]) # 'b'
#-------------------------------------------------------------------------------

def bn(x):
    return x
    # borrowed from:
    # https://github.com/ry/tensorflow-resnet/blob/master/resnet.py

    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.training import moving_averages

    BN_DECAY = 0.99 # moving average constant
    BN_EPSILON = 1e-4

    x_shape = x.get_shape() # [N,H,W,C]
    params_shape = x_shape[-1:] # [C]

    axis = list(range(len(x_shape) - 1)) # = range(3) = [0,1,2]
    # axes to reduce mean and variance.
    # apparently mean and variance is estimated per channel(feature map).

    beta = tf.Variable(tf.constant(0.,shape=params_shape),name='beta')
    gamma = tf.Variable(tf.constant(1.,shape=params_shape),name='gamma')

    moving_mean = tf.Variable(
    tf.constant(0.,shape=params_shape),name='moving_mean',trainable=False)
    moving_variance = tf.Variable(
    tf.constant(1.,shape=params_shape),name='moving_variance',trainable=False)

    def addcoll(cn,node):
        tf.add_to_collection(cn,node)

    addcoll('bn_variables', moving_mean)
    addcoll('bn_variables', moving_variance)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    # reduced mean and var(of activations) of each channel.

    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)

    addcoll('bn_updates', update_moving_mean)
    addcoll('bn_updates', update_moving_variance)

    # actual mean and var used for inference.
    mean, variance = tf.cond(get_training_mode(), # if training
        lambda: (mean, variance),
        # use immediate when training(speedup convergence)
        lambda: (moving_mean, moving_variance))
        # use average when testing(stabilize)

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x

def resconv(i,nip,nop,std=1):
    usebias = True
    nop4 = int(nop/4)
    if nip==nop and std==1:
        ident = i
        i = bn(i)
        i = relu(i)
        i = conv2d(nip,nop4,1,std=std,usebias=usebias)(i)
        i = bn(i)
        i = relu(i)
        i = conv2d(nop4,nop4,3,usebias=usebias)(i)
        i = bn(i)
        i = relu(i)
        i = conv2d(nop4,nop,1,usebias=usebias)(i)

        out = ident+i
    else:
        i = bn(i)
        i = relu(i)
        ident = i

        i = conv2d(nip,nop4,1,std=std,usebias=usebias)(i)
        i = bn(i)
        i = relu(i)
        i = conv2d(nop4,nop4,3,usebias=usebias)(i)
        i = bn(i)
        i = relu(i)
        i = conv2d(nop4,nop,1,usebias=usebias)(i)

        ident = conv2d(nip,nop,1,std=std,usebias=usebias)(ident)
        out = ident+i
    return out

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

def summary_scope(scope_name):
    var_list = get_variables_of_scope(
    tf.GraphKeys.TRAINABLE_VARIABLES,scope_name)
    print('collection: trainable_variables')
    variables_summary(var_list)

    var_list = get_variables_of_scope('bn_variables',scope_name)
    if len(var_list)>0:
        print('collection: bn_variables')
        variables_summary(var_list)

def get_variables_of_scope(collection_name,scope_name):
    var_list = tf.get_collection(collection_name, scope=scope_name)
    return var_list

def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(name)

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    names = [v.name for v in var_list]

    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6} {}'.format(
        shape_lists[i],num_elements[i],names[i]))

    print('{:>25}  ->  {:<6} {}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables),
    'variables'))

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
    def __init__(self,inputs,outputs,gt=None,sess=None,is_training=None):
        self.inputs = inputs
        self.outputs = outputs

        self.acc = None
        self.loss = None
        self.sess = sess # default none
        self.gt = gt

        self.train_step = None
        self.reporter = None
        self.is_training = is_training

        self.train_ready = False

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

    def prepare_to_train(self):
        if self.train_step is None: #if train_step not provided
            self.set_training_operation() # default training operation

        # to execute per step
        self.execution = [self.loss]
        if self.acc is not None:
            self.execution += [self.acc]

        # batch_normalization update operations
        self.auxillary_update_ops = tf.get_collection('bn_updates')

        if self.sess is None:
            self.sess = get_session()

        self.sess.run(tf.global_variables_initializer())

        self.train_ready = True

    def one_step(self,inputs,gt=None,train=False):
        if self.train_ready == False:
            self.prepare_to_train()

        # train the model on one datum/minibatch,
        # evaluate metrics, update parameters

        want_to_execute = self.execution

        if train == True:
            train_steps = [self.auxillary_update_ops, self.train_step]
            # contain all training ops into one list
            want_to_execute = [train_steps] + want_to_execute

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

    def get_epoch_runner(self,xtrain,ytrain=None,xtest=None,ytest=None):
        def r(ep=100,bs=128):
            set_training_mode(True)# flag on

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
                    # print(res)
                    message = self.step_result_report(res)

                    bar.update(min(j+batch_size,length),
                    msg = message
                    )

                bar.finish()
                timer.epend_report(length)

                if xtest is not None and ytest is not None:
                    # test set
                    set_training_mode(False) # flag off

                    timer.epstart()
                    res = self.one_step(xtest,ytest,train=False)
                    timer.epend_report(len(xtest))
                    message = self.step_result_report(res)
                    print(message)

                    set_training_mode(True) # flag on

            set_training_mode(False) # flag off
        return r

class AdvancedModelRunner:
    def check_things(self):
        for i in ['optimizer','model']:
            if not hasattr(self,i):
                raise NameError('check_things() failed:',i)
        print('things seem fine')

    def data_preloader_gen(self,nparray):
        # generate in-memory variables for datasets.
        initializer = tf.placeholder(dtype=nparray.dtype,shape=nparray.shape)
        var = tf.Variable(initializer,trainable=False,collections=[])
        return var,initializer

    def epoch_runner_preload(self,xtrain,ytrain=None,xtest=None,ytest=None):
        self.check_things()

        batch_size = 50
        num_epochs = 2
        # assume all of em are valid.
        with tf.name_scope('input_feeder'):
            xtrain_var,xtrain_init = self.data_preloader_gen(xtrain)
            ytrain_var,ytrain_init = self.data_preloader_gen(ytrain)

            # slice [N,X] into [X]
            xtrain_piece, ytrain_piece = tf.train.slice_input_producer(
            [xtrain_var, ytrain_var],
            num_epochs=None, # dont raise that stupid OORE
            shuffle=False,
            capacity=batch_size*16
            )
            # generate feed by slicing training data variables.
            # repeat for 1 epoch

            # combine [X] into [BS,X]
            xtrain_batch, ytrain_batch = tf.train.batch(
            [xtrain_piece, ytrain_piece],
            enqueue_many=False, # each feed is one single example
            capacity=batch_size*16,
            num_threads=1,
            batch_size=batch_size)
            # generate batches from feed.

        y_infer = self.model(xtrain_batch)

        loss = categorical_cross_entropy(y_infer,ytrain_batch)
        gradloss = self.optimizer.compute_gradients(loss)
        train_op = self.optimizer.apply_gradients(gradloss)

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        print('init_op...')
        sess.run(init_op)
        print('load dataset into memory...')
        sess.run(xtrain_var.initializer,feed_dict={xtrain_init: xtrain})
        sess.run(ytrain_var.initializer,feed_dict={ytrain_init: ytrain})
        # loaded into memory...

        print('init coordinator...')
        coord = tf.train.Coordinator()
        print('starting queue runners...')
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            epoch_length = len(xtrain)
            steps = int(epoch_length * num_epochs / batch_size)
            start_time = time.time()
            for i in range(steps):

                res = sess.run([train_op,loss])
                res = res[1:]
                loss_value = res[0]

                if i%100==0:
                    duration = time.time() - start_time

                    print('{} step {:6.2f} sec, {:6.2f}/sec, loss:{:6.4f}'.format(
                    i,duration,100*batch_size/duration,loss_value
                    ))

                    start_time += duration

        except tf.errors.OutOfRangeError:
            print('OORE excepted, done?')

        finally:
            coord.request_stop()
            print('stop requested')

        print('joining...')
        coord.join(threads) # wait for threads to finish
        print('done.')
