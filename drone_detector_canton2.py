from __future__ import print_function
import numpy as np

import tensorflow as tf
import canton as ct
from canton import *
import cv2

# from drone_samples_reload import load_dataset
# don't load from file, but generate on the fly
from drone_samples_generation import generate

def predetector():
    c = Can() # [N*T H W C]
    c.add(Conv2D(3,8,k=5,std=2,padding='VALID')) # 128 - 4 / 2 = 62
    c.add(Act('lrelu'))
    c.add(Conv2D(8,12,k=5,std=2,padding='VALID')) # 62 - 4 / 2 = 29
    c.add(Act('lrelu'))
    c.add(Conv2D(12,16,k=5,std=1,padding='VALID')) # 29 - 4 = 25
    c.add(Act('lrelu'))
    c.chain()
    return c

def postdetector():
    return GRUConv2D(16,8,k=5,rate=2)

pre_det = predetector()
post_det = postdetector()

def trainable_detector(): # use this to train final network (with GRU)
    c = Can()
    pd = c.add(pre_det)
    pod = c.add(post_det)
    fc = c.add(LastDimDense(8,1))

    def call(i,starting_state=None):
        s = tf.shape(i) #[NTHWC]
        i = tf.reshape(i,shape=[s[0]*s[1],s[2],s[3],s[4]])
        i = pd(i) # predetection

        ns = tf.shape(i)
        i = tf.reshape(i,shape=[s[0],s[1],ns[1],ns[2],ns[3]])
        i = pod(i,starting_state=starting_state) # post

        t = s[1] # timesteps
        ending_state = i[:,t-1,:,:,:] # extract ending_state

        i = fc(i)
        i = Act('sigmoid')(i)

        return i, ending_state

    c.set_function(call)
    return c

def trainable_detector2(): # use this to train predetector
    c = Can()
    pd = c.add(pre_det)
    fc = c.add(Conv2D(16,1,k=1))

    def call(i):
        s = tf.shape(i) #[NTHWC]
        i = tf.reshape(i,shape=[s[0]*s[1],s[2],s[3],s[4]])
        i = pd(i) # predetection
        i = fc(i)
        ns = tf.shape(i)
        i = Act('sigmoid')(i)
        i = tf.reshape(i,shape=[s[0],s[1],ns[1],ns[2],ns[3]])

        return i
    c.set_function(call)
    return c

tec = trainable_detector()
tec.summary()

def downsample(tgt):
    # print('downsampling gt...')
    s = tgt.shape
    sdim = 32
    adim = 25
    offs = int((sdim-adim) / 2)
    tgtd = np.zeros((s[0],s[1],adim,adim,s[4]),dtype='uint8')
    for i in range(len(tgt)):
        for j in range(len(tgt[0])):
            img = tgt[i,j].astype('float32')
            img = np.minimum(cv2.blur(cv2.blur(img,(5,5)),(5,5)) * 10, 255.)
            img = cv2.resize(img,dsize=(sdim,sdim),interpolation=cv2.INTER_LINEAR)
            tgtd[i,j,:,:,0] = img[offs:offs+adim,offs:offs+adim].astype('uint8')
    # print('downsampling complete.')
    return tgtd

def trainer():
    x,gt = ct.ph([None,None,None,3]), ct.ph([None,None,None,1])
    xf,gtf = tf.cast(x,tf.float32)/255.-.5,tf.cast(gt,tf.float32)/255.,

    # s = tf.shape(gtf)
    # gtf = tf.reshape(gtf,[s[0]*s[1],s[2],s[3],s[4]])
    # gtf = MaxPool2D(k=4,std=4)(gtf) # 96->24
    # ns = tf.shape(gtf)
    # gtf = tf.reshape(gtf,[s[0],s[1],ns[1],ns[2],ns[3]])

    xf += tf.random_normal(tf.shape(xf),stddev=0.05)

    y, _ending_state = tec(xf)
    loss = ct.binary_cross_entropy_loss(y,gtf,l=0.3) # bias against black
    lr = tf.Variable(1e-3)

    print('connecting optimizer...')
    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(loss,var_list=tec.get_weights())

    def feed(xin,yin,ilr):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={x:xin,gt:yin,lr:ilr})
        return res[1] # loss

    #tf.placeholder(tf.float32, shape=[None, None])
    starting_state = ct.ph([None,None,8]) # an image of some sort
    stateful_y, ending_state = tec(xf,starting_state=starting_state)

    def stateful_predict(st,i):
        # stateful, to enable fast generation.
        sess = ct.get_session()

        if st is None: # if starting state not exist yet
            res = sess.run([y,_ending_state],
                feed_dict={x:i})
        else:
            res = sess.run([stateful_y,ending_state],
                feed_dict={x:i,starting_state:st})
        return res

    return feed,stateful_predict


# async mechanism to generate samples.
import time
from collections import deque
import threading as th

sampleque = deque()
samplethread = None
def sampleloop():
    # generate samples, then put them into the que, over and over again
    while True:
        if len(sampleque)<500:
            timg,tgt = generate(1)
            tgtd = downsample(tgt)
            xt,yt = timg[0],tgtd[0]
            sampleque.appendleft((xt,yt))
        else:
            break
def needsamples(count):
    # generate our own set of samples from scratch
    # timg,tgt = generate(count)
    # tgtd = downsample(tgt)
    # xt,yt = timg,tgtd

    global samplethread
    if samplethread is None:
        samplethread = th.Thread(target=sampleloop)
        samplethread.start()
    if not samplethread.is_alive():
        samplethread = th.Thread(target=sampleloop)
        samplethread.start()

    xt,yt = [],[]
    while True:
        if len(xt)==count:
            break
        if len(sampleque)>0:
            x,y = sampleque.pop()
            xt.append(x)
            yt.append(y)
        else:
            time.sleep(.1)

    xt = np.stack(xt,axis=0)
    yt = np.stack(yt,axis=0)
    # print('generation done.')
    return xt,yt
# end async mechanism.

def r(ep=10,lr=1e-3):
    for i in range(ep):
        print('ep',i)

        # generate our own set of samples from scratch
        xt,yt = needsamples(200)

        bs = 20
        for j in range(len(xt)//bs):
            mbx = xt[j*bs:(j+1)*bs]
            mby = yt[j*bs:(j+1)*bs]
            loss = feed(mbx,mby,lr)
            print(j*bs,'loss:',loss)
        show()

def show(): # evaluate result on validation set
    from cv2tools import filt,vis
    import cv2

    # generate our own set of samples from scratch
    xt,yt = needsamples(1)

    index = np.random.choice(len(xt))
    mbx = xt[index:index+1]
    mby = yt[index:index+1]

    gru_state = None
    resarr = []
    for i in range(len(mbx[0])): # timesteps
        resy,state = stateful_predict(gru_state, mbx[0:1,i:i+1])
        resarr.append(resy) # [1,1,h,w,1]
        gru_state = state

    resarr = np.concatenate(resarr,axis=1)

    print(resarr.shape)

    vis.show_batch_autoscaled(mbx[0],name='input image')
    vis.show_batch_autoscaled(resarr[0],name='inference')
    vis.show_batch_autoscaled(mby[0],name='ground truth')

if __name__ == '__main__':
    # timg,tgt = load_dataset('drone_dataset_96x96')
    # tgtd = downsample(tgt)
    # tgtd = tgt

    feed,stateful_predict = trainer()
    ct.get_session().run(ct.gvi()) # global init
    print('ready. enter r() to train, show() to test.')
