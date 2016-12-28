import math
import keras

def lr(epoch,maxlr,minlr,t0=5,tm=2):
    tz=t0
    tcur=0
    for i in range(epoch):
        if tcur>=tz:
            tcur=0
            tz=int(tz*tm)
        else:
            tcur+=1
    nowlr = minlr+0.5*(maxlr-minlr)*(1+math.cos(float(tcur)/tz*math.pi))
    print('lr:{:.6f}, @ep {}, phase:{}/{}'.format(nowlr,epoch,tcur,tz))
    return nowlr

def gen_scheduler(minlr=1e-4,maxlr=0.05,t0=5,tm=2): # https://arxiv.org/pdf/1608.03983.pdf
    print('generating SGDR: {}<lr<{}, t0={}, tm={}'.format(minlr,maxlr,t0,tm))
    def get_lr(epoch):
        return lr(epoch,maxlr,minlr,t0,tm)

    return keras.callbacks.LearningRateScheduler(get_lr)
