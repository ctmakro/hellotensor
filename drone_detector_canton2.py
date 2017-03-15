from __future__ import print_function
import numpy as np

import tensorflow as tf
import canton as ct
from canton import *
import cv2
from drone_samples_reload import load_dataset

timg,tgt = load_dataset('drone_dataset_96x96')

def predetector():
    c = Can() # [N*T H W C]
    c.add(Conv2D(3,16,k=5,std=2)) # 48
    c.add(Act('lrelu'))
    c.add(Conv2D(16,16,k=5,std=2)) # 24
    c.add(Act('lrelu'))
    c.chain()
    return c

def postdetector():
    c = Can()
    c.add(GRUConv2D(16,32))
    c.chain()
    return c

pre_det = predetector()
post_det = postdetector()

def trainable_detector():
    c = Can()
    pd = c.add(pre_det)
    pod = c.add(post_det)
    fc = c.add(LastDimDense(32,1))

    def call(i):
        s = tf.shape(i) #[NTHWC]
        i = tf.reshape(i,shape=[s[0]*s[1],s[2],s[3],s[4]])
        i = pd(i) # predetection
        ns = tf.shape(i)
        i = tf.reshape(i,shape=[s[0],s[1],ns[1],ns[2],ns[3]])

        i = pod(i)
        i = fc(i)
        i = Act('sigmoid')(i)
        return i
    c.set_function(call)
    return c

tec = trainable_detector()
tec.summary()

def downsample(tgt):
    print('downsampling gt...')
    s = tgt.shape
    sdim = 24
    tgtd = np.zeros((s[0],s[1],sdim,sdim,s[4]),dtype='uint8')
    for i in range(len(tgt)):
        for j in range(len(tgt[0])):
            img = tgt[i,j]
            tgtd[i,j,:,:,0] = cv2.resize(img,dsize=(sdim,sdim),interpolation=cv2.INTER_LINEAR)
    print('downsampling complete.')
    return tgtd

# tgtd = downsample(tgt)
tgtd = tgt

def trainer():
    x,gt = ct.ph([None,None,None,3]), ct.ph([None,None,None,1])
    xf,gtf = tf.cast(x,tf.float32)/255.,tf.cast(gt,tf.float32)/255.,

    s = tf.shape(gtf)
    gtf = tf.reshape(gtf,[s[0]*s[1],s[2],s[3],s[4]])
    gtf = MaxPool2D(k=4,std=4)(gtf) # 96->24
    ns = tf.shape(gtf)
    gtf = tf.reshape(gtf,[s[0],s[1],ns[1],ns[2],ns[3]])

    y = tec(xf)
    loss = ct.binary_cross_entropy_loss(y,gtf)
    lr = tf.Variable(1e-3)

    print('connecting optimizer...')
    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(loss,var_list=tec.get_weights())

    def feed(xin,yin,ilr):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={x:xin,gt:yin,lr:ilr})
        return res[1] # loss
    return feed

feed = trainer()
ct.get_session().run(ct.gvi()) # global init

xt = timg
yt = tgtd
def r(ep=10,lr=1e-3):
    for i in range(ep):
        print('ep',i)
        bs = 5
        for j in range(len(xt)//bs):
            mbx = xt[j*bs:(j+1)*bs]
            mby = yt[j*bs:(j+1)*bs]
            loss = feed(mbx,mby,lr)
            print(j*bs,'loss:',loss)

def show(): # evaluate result on validation set
    from cv2tools import filt,vis
    import cv2

    index = np.random.choice(len(xt))
    mbx = xt[index:index+1]
    mby = yt[index:index+1]

    result = tec.infer(mbx)
    print(result.shape)

    vis.show_batch_autoscaled(mbx[0],name='input image')
    vis.show_batch_autoscaled(result[0],name='inference')
    vis.show_batch_autoscaled(mby[0],name='ground truth')
