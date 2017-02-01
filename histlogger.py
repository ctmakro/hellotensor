from __future__ import print_function

from keras.utils import np_utils
import keras
from keras import backend as K
import math

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time

Cb=keras.callbacks.Callback

class EpochEndCallback(Cb):
    def __init__(self,f):
        super(Cb, self).__init__()
        self.f = f
    def on_epoch_end(self, epoch, logs={}):
        self.f()

class LoggerCallback(Cb):
    def __init__(self,keys=None,autoplot=True,interval=3.):
        super(Cb, self).__init__()

        if keys is None:
            keys = [
            {'loss':[],'val_loss':[]},
            {'acc':[],'val_acc':[]}
            ]

        self.losshist = keys
        self.timestamps = []

        self.autoplot = autoplot
        self.start_time=time.time()
        self.update_timer = time.time()
        self.update_interval=interval
        self.plot_inited = False

    def on_epoch_end(self, epoch, logs={}):
        if self.plot_inited==False:
            self.initplot()
            self.plot_inited=True


        lr=K.get_value(self.model.optimizer.lr)
        logs['lr'] = lr

        if self.autoplot:
            t = time.time() - self.start_time

            losshist = self.losshist
            for s in losshist:
                for k in s:
                    lk = logs[k]
                    s[k].append(max(lk,1e-10))
                    if len(s[k])>1000:
                        s[k].pop(0)

            self.timestamps.append(t)
            if len(self.timestamps)>1000:
                self.timestamps.pop(0)

            if time.time() > self.update_timer + self.update_interval:
                self.update_timer = time.time()
                self.updateplot()

    def on_batch_end(self, batch, logs={}):
        pass
        # plt.pause(0.001)

    def initplot(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        plt.pause(0.001)

    def updateplot(self):
        fig = self.fig
        ax = self.ax
        ax2 = self.ax2
        timestamps = self.timestamps

        ax.clear()
        ax2.clear()
        ax.grid(True)
        ax2.grid(True)
        ax.set_yscale('log')
        #ax2 keeps lin
        losshist = self.losshist
        for s in losshist:
            for k in s:
                if len(k.split('loss'))>1:
                    ax.plot(timestamps,s[k],label=k)
                else:
                    # non-loss goes linear
                    ax2.plot(timestamps,s[k],label=k)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper left')
        #plt.draw()
        fig.canvas.draw()
        plt.pause(0.001)
