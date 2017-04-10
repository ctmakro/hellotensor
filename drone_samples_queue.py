import cv2
import numpy as np

# from drone_samples_reload import load_dataset
# don't load from file, but generate on the fly
from drone_samples_generation import generate

def downsample(tgt):
    # print('downsampling gt...')
    s = tgt.shape
    sdim = 32
    adim = 9
    offs = int((sdim-adim) / 2)
    tgtd = np.zeros((s[0],s[1],adim,adim,s[4]),dtype='uint8')
    for i in range(len(tgt)):
        for j in range(len(tgt[0])):
            img = tgt[i,j].astype('float32')
            img = np.minimum(cv2.blur(cv2.blur(img,(3,3)),(3,3)) * 10, 255.)
            img = cv2.resize(img,dsize=(sdim,sdim),interpolation=cv2.INTER_LINEAR)
            tgtd[i,j,:,:,0] = img[offs:offs+adim,offs:offs+adim].astype('uint8')
    # print('downsampling complete.')
    return tgtd

# async mechanism to generate samples.
import time
from collections import deque
import threading as th

sampleque = deque()
samplethread = [None,None]
def sampleloop():
    # generate samples, then put them into the que, over and over again
    while True:
        if len(sampleque)<500:
            timg,tgt = generate(num_track=1,num_per_track=1)
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
    for i in range(len(samplethread)):
        if samplethread[i] is None:
            samplethread[i] = th.Thread(target=sampleloop)
            samplethread[i].start()
        if not samplethread[i].is_alive():
            samplethread[i] = th.Thread(target=sampleloop)
            samplethread[i].start()

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
