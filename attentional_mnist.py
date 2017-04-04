from __future__ import print_function

import tensorflow as tf
import numpy as np
import canton as ct
from canton import *

def mnist_data():
    print('loading mnist...')
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    # reshape into 2d
    # X_train = X_train.reshape(X_train.shape[0],784)
    # X_test = X_test.reshape(X_test.shape[0],784)

    X_train.shape += 1,
    X_test.shape += 1,

    print('X_train shape:', X_train.shape, X_train.dtype)
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

xt,yt,xv,yv = mnist_data()

def castf32(i):
    return tf.cast(i,tf.float32)

# RBF glimpse
# evaluate RBF functions representing foveal attention mechanism over the input image, given offset.
class Glimpse2D(Can):
    def __init__(self, num_receptors, pixel_span=20):
        super().__init__()
        if num_receptors<1:
            raise NameError('num_receptors should be greater than 0')
        self.num_receptors = nr = num_receptors
        self.pixel_span = ps = pixel_span

        # generate initial positions for receptive fields
        positions = np.zeros((nr,2),dtype='float32')
        w = int(np.ceil(np.sqrt(nr)))
        index = 0
        for row in range(w):
            for col in range(w):
                if index<nr:
                    positions[index,0] = row/(w-1)
                    positions[index,1] = col/(w-1)
                    index+=1
                else:
                    break

        # positions = np.random.uniform(low=-ps/2,high=ps/2,size=(nr,2)).astype('float32')
        positions = (positions - 0.5) * ps * 0.5
        m = tf.Variable(positions,name='means')
        self.weights.append(m)
        self.means = m

        # stddev of receptive fields
        stddevs = (np.ones((nr,1))*ps*0.2).astype('float32')
        s = tf.Variable(stddevs,name='stddevs')
        self.weights.append(s)
        self.stddevs = s

    def shifted_means_given_offsets(self,offsets):
        means = self.means # [num_of_receptor, 2]

        means = tf.expand_dims(means,axis=0) # [batch, num_of_receptor, 2]
        offsets = tf.expand_dims(offsets,axis=1) # [batch, num_of_receptor, 2]

        shifted_means = means + offsets # [batch, num_of_receptor, 2]

        return shifted_means

    def variances(self):
        variances = tf.nn.softplus(self.stddevs)**2 # [num_of_receptor, 1]
        return variances

    def __call__(self,i): # input: [image, offsets]
        offsets = i[1] # offsets [batch, 2]
        images = i[0] # [batch, h, w, c]

        shifted_means =\
            self.shifted_means_given_offsets(offsets)

        variances = self.variances() # [num_of_receptor, 1]

        ish = tf.shape(images) # [batch, h, w, c]
        # UVMap, aka coordinate system
        u,v = tf.range(start=0,limit=ish[1],dtype=tf.int32),\
                tf.range(start=0,limit=ish[2],dtype=tf.int32)
        # U, V -> [hpixels], [wpixels]

        u,v = castf32(u) - (castf32(ish[1])-1)/2, \
            castf32(u) - (castf32(ish[2])-1)/2

        u = tf.expand_dims(u, axis=0)
        u = tf.expand_dims(u, axis=0)
        u = tf.expand_dims(u, axis=3)

        v = tf.expand_dims(v, axis=0)
        v = tf.expand_dims(v, axis=0)
        v = tf.expand_dims(v, axis=0)
        # U, V -> [1, 1, hpixels, 1], [1, 1, 1, wpixels]
        # where hpixels = [-0.5...0.5] * image_height
        # where wpixels = [-0.5...0.5] * image_width

        smh = tf.expand_dims(shifted_means[:,:,0:1], axis=2)
        # [batch, num_of_receptor, 1(h)]
        smw = tf.expand_dims(shifted_means[:,:,1:2], axis=3)
        # [batch, num_of_receptor, 1(w)]

        # RBF that sum to one over entire x-y plane:
        # integrate
        #   e^(-((x-0.1)^2+(y-0.3)^2)/v) / (v*pi)
        #   dx dy x=-inf to inf, y=-inf to inf, v>0
        # where ((x-0.1)^2+(y-0.3)^2) is the squared distance on the 2D plane

        squared_dist = (smh - u)**2 + (smw - v)**2
        # [batch, num_of_receptor, hpixels, wpixels]

        variances = tf.expand_dims(variances, axis=0)
        # [1, num_of_receptor, var]
        variances = tf.expand_dims(variances, axis=2)
        # [1, num_of_receptor, 1, var]

        density = tf.exp(- squared_dist / variances) / \
                (variances * np.pi)
        # [b, n, h, w] / [1, n, 1, 1]
        # should sum to 1

        density = tf.expand_dims(density, axis=4)
        # [b, n, h, w, 1]

        images = tf.expand_dims(images, axis=1)
        # [b, h, w, c] -> [b, 1, h, w, c]

        responses = tf.reduce_sum(density * images, axis=[2,3])
        # [batch, num_of_receptor, channel]
        return responses

class GRU_Glimpse2D_onepass(Can):
    def __init__(self, num_h, num_receptors, channels, pixel_span=20):
        super().__init__()

        self.channels = channels # explicit
        self.num_h = num_h
        self.num_receptors = num_receptors
        self.pixel_span = pixel_span # how far can the fovea go

        num_in = channels * num_receptors

        self.glimpse2d = g2d = Glimpse2D(num_receptors, pixel_span)
        self.gru_onepass = gop = ct.cans.GRU_onepass(num_in,num_h)
        self.hidden2offset = h2o = Dense(num_h,2)
        # self.glimpse2gru = g2g = Dense(num_in,num_gru_in)

        self.incan([g2d,gop,h2o,g2g])

    def __call__(self,i):
        hidden = i[0] # hidden state of gru [batch, dims]
        images = i[1] # input image [NHWC]

        g2d = self.glimpse2d
        # g2g = self.glimpse2gru
        gop = self.gru_onepass
        h2o = self.hidden2offset

        # hidden is of shape [batch, dims], range [-1,1]
        offsets = self.get_offset(hidden) # [batch, 2]

        responses = g2d([images,offsets]) # [batch, num_receptors, channels]
        rsh = tf.shape(responses)
        responses = tf.reshape(responses,shape=(rsh[0],rsh[1]*rsh[2]))

        # responses2 = g2g(responses)
        # responses2 = Act('lrelu')(responses2)
        hidden_new = gop([hidden,responses])
        return hidden_new

    def get_offset(self, hidden):
        # given hidden state of GRU, calculate next step offset
        # hidden is of shape [batch, dims], range [-1,1]
        h2o = self.hidden2offset
        offsets = tf.tanh(h2o(hidden)) # [batch, 2]
        offsets = offsets * self.pixel_span / 2
        return offsets

# def glimpse2dtest():
#     g2d = Glimpse2D(1)
#
#     offsets = np.array([[0,0],[2,2]],dtype='float32')
#     images = np.ones((1,32,32,1),dtype='float32')
#
#     pmvi = [images,offsets]
#     pmvit = [tf.Variable(p) for p in pmvi]
#
#     for t in pmvi:
#         print(t.shape)
#
#     resp = g2d(pmvit)
#
#     sess = get_session()
#     sess.run(gvi())
#     res = sess.run(resp)
#
#     print(res)
#     print(res.shape)

# glimpse2dtest()

GRU_Glimpse2D = rnn_gen('GG2D', GRU_Glimpse2D_onepass)
gg2d = GRU_Glimpse2D(num_h=64, num_receptors=8, channels=1, pixel_span=28)

def classifier():
    c = Can()
    g = c.add(gg2d)
    d1 = c.add(TimeDistributedDense(64,10))
    c.unit = g.unit # rnn onepass instance
    def call(i,return_hidden_states=False):
        # input should be image sequence: [NTHWC]
        hidden_states = g(i,state_shaper=lambda inp,hid:[tf.shape(inp)[0],hid])
        # state_shaper is used to determine the shape of the initial state
        # for rnn
        # i -> [batch, timesteps, hidden]

        i = d1(hidden_states)
        if not get_training_state():
            i = Act('softmax')(i) # apply softmax only while not training

        if return_hidden_states:
            return i, hidden_states
        else:
            return i
    c.set_function(call)
    return c

gg2dclf = classifier()
gg2dclf.summary()

def trainer():
    inp = ct.ph([None,None,1]) # image
    gt = ct.ph([10]) # labels

    x = inp-0.5
    x = tf.expand_dims(x,axis=1) #[NHWC] -> [N1HWC]
    gt2 = tf.expand_dims(gt,axis=1) #[batch, dims] -> [batch, 1, dims]
    timesteps = 9 # how many timesteps would you evaluate the RNN

    x = tf.tile(x,multiples=[1,timesteps,1,1,1])
    gt2 = tf.tile(gt2,multiples=[1,timesteps,1])

    y = gg2dclf(x) # [batch, timesteps, 10]

    loss = mean_softmax_cross_entropy(y, gt2)
    # mean of cross entropy, over all timesteps.

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss, var_list=gg2dclf.get_weights())

    def feed(img,lbl):
        sess = get_session()
        res = sess.run([train_step,loss],feed_dict={
            inp:img, gt:lbl
        })
        return res[1]

    # extract foveal pattern from hidden states
    set_training_state(False) # set training state to false to enable softmax
    y_softmaxed,hiddens = gg2dclf(x,return_hidden_states=True)
    # [batch, timesteps, 10], [batch, num_h]
    set_training_state(True)

    hs = tf.shape(hiddens)
    hiddens = tf.reshape(hiddens,shape=[-1,hs[2]]) #[batch*time, dims]

    offsets = gg2dclf.unit.get_offset(hiddens)

    shifted_means = gg2dclf.unit.glimpse2d.shifted_means_given_offsets(offsets)
    shifted_means = tf.reshape(shifted_means,shape=[hs[0],hs[1],-1,2]) #[batch*time,num_receptors,2] -> [batch,time,num_receptors,2]

    variances = gg2dclf.unit.glimpse2d.variances() #[num_receptors,1]

    def test(img):
        sess = get_session()
        res = sess.run([x,y_softmaxed,shifted_means,variances],feed_dict={
            inp:img
        })
        return res

    return feed,test

feed,test = trainer()

get_session().run(gvi())

def r(ep=10):
    length = len(xt)
    bs = 20
    for i in range(ep):
        print('ep:',i)
        for j in range(0,length,bs):
            mbx = xt[j:j+bs]
            mby = yt[j:j+bs]
            loss = feed(mbx,mby)
            print(j,'loss:',loss)
            if j% 200 == 0:
                show()

def show():
    import cv2
    from cv2tools import vis,filt

    index = np.random.choice(len(xt))
    img = xt[index:index+1]
    tiledx,y_softmaxed,shifted_means,variances = test(img)

    for idxt, dist in enumerate(y_softmaxed[0]):
        print('step',idxt,'guess:',np.argmax(dist))

    tiledx += 0.5
    tiledx_copy = tiledx.copy()
    tiledx = (tiledx*255.).astype('uint16') # 16-bit-ify
    tiledx = np.tile(tiledx,(1,1,1,3)) # colorify

    shifted_means += np.array([img.shape[1]-1,img.shape[2]-1],dtype='float32')/2
    # shift from image center to image coordinates

    # draw the circles...
    for idxt,receptors in enumerate(shifted_means[0]):
        tmp = tiledx[0,idxt]*0 # [HWC]
        for idxr,receptor in enumerate(receptors):
            tmp += cv2.circle(np.zeros_like(tmp,dtype='uint8'), (int(receptor[1]*16), int(receptor[0]*16)),
                radius=int(np.sqrt(variances[idxr,0])*16),
                color=(80,140,180), thickness=-1,
                lineType=cv2.LINE_AA, shift=4)

        tiledx[0,idxt] = tiledx[0,idxt]*0.5 + tmp*0.5

    tiledx = tiledx.clip(min=0,max=255).astype('uint8')
    vis.show_batch_autoscaled(tiledx_copy[0],name='input sequence')
    vis.show_batch_autoscaled(tiledx[0],name='attention over time')
