# some cv2 shit.

import cv2
import numpy as np
import tensorflow as tf
import time
import canton as ct
from canton import *

from drone_detector_canton2 import pre_det, post_det, tec, trainer

feed,stateful_predict = trainer()
ct.get_session().run(ct.gvi()) # global init

tec.load_weights('tec.npy')

def mainloop():
    print('starting video...')
    cap = cv2.VideoCapture(0)

    lastelapsed = .1
    gru_state = None # hidden state for ConvGRU

    while(True):
        ret,frame = cap.read()
        starttime = time.time()
        if ret != True:
            print(ret)
            break

        # preprocessing
        frame = cv2.resize(frame,dsize=(320,240))

        # inference
        fi = frame.view()
        fi.shape = (1,1)+frame.shape
        resy,state = stateful_predict(gru_state, fi)

        resy.shape = resy.shape[2:]
        gru_state = state*.9

        # timestamping
        elapsed = time.time()-starttime
        elapsed = elapsed*.1+lastelapsed*.9
        cv2.putText(frame, "{} ms".format(int(elapsed*1000)),
    		(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        lastelapsed = elapsed

        # post processing
        eff = (resy*255.).astype('uint8')
        # eff = cv2.medianBlur(eff,3)
        eff = cv2.applyColorMap(eff,cv2.COLORMAP_HOT)
        eff = cv2.resize(eff,dsize=(eff.shape[1]*4,eff.shape[0]*4),
            interpolation=cv2.INTER_NEAREST)

        # blend and display
        frame = frame.astype('uint16')
        xborder = (frame.shape[1] - eff.shape[1])//2
        yborder = (frame.shape[0] - eff.shape[0])//2
        frame[yborder:yborder+eff.shape[0],xborder:xborder+eff.shape[1]]+=eff
        frame = np.clip(frame,a_max=255,a_min=0)
        frame = frame.astype('uint8')
        cv2.imshow('capture',frame)

        # cv2.imshow('effect',eff)
        # cv2.imshow('prediction',heat)

        if cv2.waitKey(1)& 0xff == ord('q'):
            break # if q is the pressed key, then exit loop

    cap.release()
    cv2.destroyAllWindows()

mainloop()
