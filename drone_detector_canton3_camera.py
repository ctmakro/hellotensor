# some cv2 shit.

import cv2
import numpy as np
import tensorflow as tf
import time
import canton as ct
from canton import *
from cv2tools import vis,filt

from lucas_kanade import app
from drone_detector_canton3 import classifier

clf = classifier()
get_session().run(ct.gvi()) # global init

clf.load_weights('clf.npz')

def pyr(i,scale):
    j = i.copy()
    if scale>0:
        for k in range(scale):
            j = cv2.pyrUp(j)
    elif scale<0:
        for k in range(-scale):
            j = cv2.pyrDown(j)
    else:
        j = j
    return j

class theapp(app):
    def process(self,lastframe,thisframe):
        # graify, downscale
        fg = thisframe.copy()

        # fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        pyramid_downscaler = 2
        pyramid = []
        for i in range(pyramid_downscaler):
            fg = pyr(fg,-1)
            pyramid.append(fg)
            # size: -1 -2 -3

        probmap = None

        for idx,img in enumerate(pyramid):
            cv2.imshow('pyr'+str(idx), img)

            # inferrence
            img.shape = (1,)+img.shape
            img = img.astype('float32')
            j = clf.infer(img/255.-0.5)[0]

            # j = j * (j>0.5)
            # vis.show_autoscaled(j[0],name='infer_pyr_'+str(idx),limit=300.)
            j = pyr(j, idx) # idx: [0 1 2]
            if j is None:
                print('j is none',img.dtype)

            if probmap is None:
                probmap = j
            else:
                if len(j.shape)==2:
                    j.shape+=(1,)
                offy = int((probmap.shape[0] - j.shape[0])//2)
                offx = int((probmap.shape[1] - j.shape[1])//2)

                probmap[offy:offy+j.shape[0],offx:offx+j.shape[1],:] += j

            # cv2.imshow('infer_pyr_1',j[0])

        # probmap = cv2.blur(probmap,(3,3))

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(probmap)
        # image = orig.copy()
        if maxVal>0.9:
            shift = 50 # a small offset since convnet cut the border off
            scaler = (thisframe.shape[1] - shift*2) / probmap.shape[1]
            # black-white ring
            cv2.circle(thisframe,
                (int(maxLoc[0]*scaler+shift),int(maxLoc[1]*scaler+shift)),
                30, (0, 0, 0), 4)
            cv2.circle(thisframe,
                (int(maxLoc[0]*scaler+shift),int(maxLoc[1]*scaler+shift)),
                30, (255, 255, 255), 2)

        vis.show_autoscaled(probmap,name='scaled_back',limit=300.)

        return thisframe

if __name__ == '__main__':
    inst = theapp()
    inst.loop()
