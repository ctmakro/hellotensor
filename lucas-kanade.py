# effectively detect motion between frames using lucas-kanade.

import cv2
import numpy as np
# import tensorflow as tf
import time
# import canton as ct
# from canton import *

class track:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.age = 0

    def update(self,x,y):
        self.x = x
        self.y = y
        self.age+=1

class app:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.lastelapsed = .1
        self.lastframe = None

        # params for ShiTomasi corner detection
        self.feature_params = dict(
            qualityLevel = 0.05,
            minDistance = 7,
            blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(
            winSize  = (15,15),
            maxLevel = 8,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.tracks = []
        self.oldfg = None
        print('initialized.')

    def loop(self):
        print('starting video loop...')
        cap = self.cap

        while(True):
            ret, thisframe = cap.read()
            starttime = time.time()
            if ret != True:
                print(ret)
                break

            frame = self.process(self.lastframe,thisframe)
            self.lastframe = thisframe

            # timestamping
            elapsed = time.time()-starttime
            elapsed = elapsed*.1 + self.lastelapsed*.9
            cv2.putText(frame, "{:4d} ms".format(int(elapsed*1000)),
        		(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            self.lastelapsed = elapsed
            cv2.imshow('capture',frame)

            if cv2.waitKey(1)& 0xff == ord('q'):
                break # if q is the pressed key, then exit loop
        cap.release()
        cv2.destroyAllWindows()

    def process(self,lastframe,thisframe):
        oldfg, tracks = self.oldfg, self.tracks

        # graify, downscale
        fg = thisframe

        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        pyramid_downscaler = 4
        for i in range(int(np.log2(pyramid_downscaler))):
            fg = cv2.pyrDown(fg)

        # process only when lastframe is available
        if self.lastframe is None:
            pass
        else:
            fs = fg.shape
            # kill all tracks that are too close to border
            tracks = list(filter(\
                lambda t:t.x>10 and t.x<fs[1]-10 and t.y>10 and t.y<fs[0]-10,\
                tracks))

            '''masking'''

            # paint a mask of where all the available tracks are
            mask = np.zeros_like(fg,dtype='uint8')
            mask[10:fs[0]-10,10:fs[1]-10] = 255 # black the borders: dont sample near border.
            for t in tracks:
                cv2.circle(mask, (int(t.x), int(t.y)), radius=7,
                    color=0, thickness=-1)

            cv2.imshow('mask',mask)

            '''track resampling'''

            # sample new tracks, if no enough of them
            if len(tracks)<40:
                ps = cv2.goodFeaturesToTrack(
                    fg,
                    mask = mask,
                    maxCorners = 40-len(tracks),
                    **self.feature_params)

                if ps is not None:
                    # for reasons unknown, ps looks like [[[x,y]],[[x,y]]]
                    for p in ps:
                        tracks.append(track(p[0][0],p[0][1]))

            # if there aren't enough tracks to track:
            if len(tracks)<1:
                print('no enough tracks:',len(tracks))
            else:
                '''tracking'''

                # forward and backward LK, to detect faulty tracks
                img0, img1 = oldfg, fg
                p0 = np.array([[t.x,t.y] for t in tracks])
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                good = (st[:,0] == 1)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                d = np.sum((p0 - p0r)**2, axis=1)

                good = good & (d < 1) & (st[:,0] == 1)

                # update the tracks (and kill the bad ones)
                newtracks = []
                for i in range(len(tracks)):
                    if good[i]==True:
                        t = tracks[i]
                        t.update(p1[i,0],p1[i,1]) # update coordinates
                        newtracks.append(t)
                tracks = newtracks

                '''painting'''

                # pick out the good tracks (for painting)
                # good_new = np.compress(good,p1,axis=0)
                good_old = np.compress(good,p0,axis=0)

                scaler = pyramid_downscaler
                # blue -> old, red -> new
                for p in good_old:
                    cv2.circle(thisframe,
                        (int(p[0]*scaler),int(p[1]*scaler)),
                        radius=5,
                        color=[255,255,120],thickness=1)

                for p in tracks:
                    cv2.circle(thisframe,
                        (int(p.x*scaler),int(p.y*scaler)),
                        radius=5,
                        color=[30,128,255],thickness=1)

                    cv2.putText(thisframe, "{:4d}".format(p.age),
                        (int(p.x*scaler), int(p.y*scaler)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 200), 1)

        self.oldfg = fg
        self.tracks = tracks
        cv2.imshow('fg',fg)
        return thisframe

inst = app()
inst.loop()
