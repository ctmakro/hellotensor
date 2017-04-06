from __future__ import print_function
import numpy as np

import tensorflow as tf
import canton as ct
from canton import *
import cv2

# this time trying to train a multi-stage classifier
# don't exploit motion information(!)

from drone_samples_queue import needsamples

def classifier():
    c = Can() #[NHWC]
    c1 = c.add(Conv2D(3,16,k=5,std=2,padding='VALID'))
    c2 = c.add(Conv2D(16,32,k=5,std=2,padding='VALID'))
    c3 = c.add(Conv2D(32,16,k=5,std=2,padding='VALID'))
    c4 = c.add(Conv2D(16,1,k=1,std=1,padding='VALID'))
    def call(i):
        i = c1(i)
        i = Act('relu')(i)
        i = c2(i)
        i = Act('relu')(i)
        i = c3(i)
        i = Act('relu')(i)
        i = c4(i)
        i = Act('sigmoid')(i)
        return i
    c.set_function(call)
    return c

clf = classifier()
clf.summary()

def trainer():
    x,gt = ct.ph([None,None,3]), ct.ph([None,None,1])
    xf,gtf = tf.cast(x,tf.float32)/255.-.5,tf.cast(gt,tf.float32)/255.,

    # s = tf.shape(gtf)
    # gtf = tf.reshape(gtf,[s[0]*s[1],s[2],s[3],s[4]])
    # gtf = MaxPool2D(k=4,std=4)(gtf) # 96->24
    # ns = tf.shape(gtf)
    # gtf = tf.reshape(gtf,[s[0],s[1],ns[1],ns[2],ns[3]])

    xf += tf.random_normal(tf.shape(xf),stddev=0.05)

    y = clf(xf)
    loss = ct.binary_cross_entropy_loss(y,gtf,l=2.) # bias against black
    lr = tf.Variable(1e-3)

    print('connecting optimizer...')
    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(loss,var_list=clf.get_weights())

    def feed(xin,yin,ilr):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={
            x:xin,
            gt:yin,
            lr:ilr
        })
        return res[1] # loss

    def predict(i):
        sess = ct.get_session()
        res = sess.run([y],
            feed_dict={x:i})
        return res[0]

    return feed,predict

def r(ep=10,lr=1e-3):
    for i in range(ep):
        print('ep',i)

        # generate our own set of samples from scratch
        xt,yt = needsamples(200)

        bs = 20
        for j in range(len(xt)//bs):
            mbx = xt[j*bs:(j+1)*bs]
            mby = yt[j*bs:(j+1)*bs]

            s = mbx.shape
            mbx.shape = s[0]*s[1],s[2],s[3],s[4]
            s = mby.shape
            mby.shape = s[0]*s[1],s[2],s[3],s[4]

            loss = feed(mbx,mby,lr)
            print(j*bs,'loss:',loss)
        show()

def show(): # evaluate result on validation set
    from cv2tools import filt,vis
    import cv2

    # generate our own set of samples from scratch
    mbx,mby = needsamples(10)

    mbx.shape = (10,) +mbx.shape[2:]
    mby.shape = (10,) +mby.shape[2:]

    res = predict(mbx)

    print(res.shape)

    vis.show_batch_autoscaled(mbx,name='input image')
    vis.show_batch_autoscaled(res,name='inference')
    vis.show_batch_autoscaled(mby,name='ground truth')

if __name__ == '__main__':
    # timg,tgt = load_dataset('drone_dataset_96x96')
    # tgtd = downsample(tgt)
    # tgtd = tgt

    feed,predict = trainer()
    ct.get_session().run(ct.gvi()) # global init
    print('ready. enter r() to train, show() to test.')
