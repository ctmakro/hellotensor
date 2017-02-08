from tensorflow_boilerplate import *
from tensorflow_session import get_session
import tensorflow as tf
import numpy as np

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

from waves import song,loadfile,show_waterfall,play,fs

sample_length = 1024
time_steps = 60 # about 1s
zed = 32

def bn(i):
    return BatchNormalization(axis=-1)(i)

def relu(i):
    return LeakyReLU(0.2)(i)

def dis():
    i = Input(shape=(time_steps * sample_length, 1))
    # shape: (batch, time_steps * 1024, 1)
    inp = i

    # now try to conv it into time_steps...

    def c1d(nb_filter,dim,std=1):
        def j(i):
            i = Convolution1D(nb_filter, dim,
                subsample_length=std,border_mode='same')(i)
            return i
        return j

    i = c1d(16, dim=4, std=1)(i) # 1024

    for k in range(10):
        # k = 0..9
        feat = min(2**k * 16, 64)
        i = bn(i)
        i = relu(i)
        i = c1d(feat,dim=4,std=2)(i) # 512..1

    # now shape: (batch, time_steps, 16)

    i = LSTM(32,
        input_dim=16,
        input_length=time_steps,
        return_sequences=False,
        stateful=False,
        consume_less='gpu')(i)

    # now shape: (batch, 32)

    i = relu(i)
    i = Dense(1)(i)
    i = Activation('sigmoid')(i) # score

    model = Model(input=inp,output=i)
    return model
    
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
    
def hole_strech(i,axis): # make holes in input data, eg [1,2] -> [1,0,2,0]
    def makehole(a):
        s = tf.shape(a)
        z = tf.zeros(s)
        stack = tf.stack([a,z],axis=axis)
        holed = tf.reshape(stack,[s[0],s[1]*2,s[2]])
        return holed

    def shaper(ishape):
        return (ishape[0],ishape[1]*2,ishape[2])

    return Lambda(lambda x:makehole(x),output_shape=shaper)(i)

def gen():
    i = Input(shape=(time_steps,zed))
    # shape: (batch, time_steps, zed)
    inp = i

    i = LSTM(32,
        input_dim=zed,
        input_length=time_steps,
        return_sequences=True,
        stateful=False,
        consume_less='gpu')(i)

    # shape: (batch, time_steps, 16)
    # now try to deconv it into sample_length...

    def ct1d(nb_filter,dim,std=1):
        def j(i):
            # input: batch, length, feature
            # meant: batch, length * std, nb_filter
            if std==1:
                i = i
            else:
                i = hole_strech(i,axis=1)
            i = Convolution1D(nb_filter, dim,
                border_mode='same')(i)
            return i
        return j

    # 1

    for k in reversed(range(10)): # 9..0
        feat = min(2**k * 16,64)
        
        i = ct1d(feat,4,std=2)(i) # time_steps*2
        i = bn(i)
        i = relu(i)

    # batch, time_steps * 1024, feat
   
    i = ct1d(1, 1, std=1)(i) # shape: (batch, time_steps*1024, 1)
    i = Activation('tanh')(i)

    model = Model(input=inp, output=i)
    return model

dm,gm = dis(),gen()
dm.summary()
gm.summary()

def gan(g,d):
    # initialize a GAN trainer

    # this is the fastest way to train a GAN in Keras
    # two models are updated simutaneously in one pass

    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)

    # single side label smoothing: replace 1.0 with 0.9
    dloss = - K.mean(log_eps(1-gscore) + .01 * log_eps(1-rscore) + .99 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    # learning_phase = tf.get_default_graph().get_tensor_by_name(
    #     'keras_learning_phase:0')

    learning_phase = K.learning_phase()

    def gan_feed(sess,batch_image,z_input):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data,learning_phase

        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_data:batch_image,
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed

gan_feed = gan(gm,dm)

print('Ready. enter r() to train')

floatsong = song.astype('float32') / 32767.

def r(ep=10000,noise_level=.01):
    sess = K.get_session()

    batch_size = 4
    length_example = sample_length * time_steps
    length_batch = batch_size * length_example
    srange = len(floatsong) - length_batch

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from song
        j = np.random.choice(srange)
        minibatch = floatsong[j:j+length_batch,0] # use mono
        # minibatch = np.reshape(minibatch,[batch_size,length_example,1])
        minibatch.shape = (batch_size,length_example,1)

        # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,time_steps,zed))

        # train for one step
        losses = gan_feed(sess,minibatch,z_input)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show()

def show():
    length = 1 # 32 * 32768 samples
    snd = gm.predict(np.random.normal(loc=0,scale=1,size=(length,time_steps,zed)))
    snd = snd.reshape((snd.shape[0]*snd.shape[1],))
    print('snd shape:',snd.shape)

    # following functions are prepared for 16bit signed audio
    snd16bit = (snd*32000.).astype('int32')
    play(snd16bit)
    show_waterfall(snd16bit)
