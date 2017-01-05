from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input ,Lambda
from keras.layers import Convolution2D, merge, MaxPooling2D,AtrousConvolution2D,AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D
from keras.optimizers import SGD,Adam,Adadelta,Nadam
from keras.utils import np_utils
import keras
from math import *
import keras.backend as K
import sgdr
import numpy as np
import time

from imagelayer import ImageLayer,SineLayer,MultipleInputLayer

lrelu = keras.layers.advanced_activations.LeakyReLU(0.5)
srelu = keras.layers.advanced_activations.SReLU

def torque(currents,theta):

    # currents are ampere
    # theta is radian
    # 4x3 pole

    fluxes = currents * [1.3,0.8,1.0]

    fluxes = np.tanh(fluxes*2)/2

    # poleshift
    ps0 = .51
    ps1 = .27

    # fluxes = currents

    def park(tensor,theta):
        park_matrix = np.array([
        [cos(theta+ps0),cos(ps1+theta-2./3.*pi),cos(theta+2./3.*pi)],
        [sin(theta+ps0),sin(ps1+theta-2./3.*pi),sin(theta+2./3.*pi)],
        [1./2,1./2,1./2]
        ])
        return np.dot(park_matrix,tensor.T).T * (2./3)

    dq_fluxes = park(fluxes,theta)

    return fluxes,dq_fluxes

def generate_data_pair(length,random=True,model=None):
    # r: electrical phase array
    # rt: rotor phase array
    # cs: currents array

    # r = np.random.rand(length)*2*pi

    if random:
        rt = np.random.rand(length)*4.*pi
    else:
        rt = np.arange(length).astype('float32')/length*9.*pi # 3 rev

    # currents = np.zeros(r.shape+(3,))
    # currents[:,0],currents[:,2],currents[:,1] = np.sin(r),np.sin(r+2/3*pi),np.sin(r+4/3*pi)

    if random:
        currents = np.random.normal(loc=0.0,scale=1.0,size=(length,3))
        currents[:,0:3] -= np.mean(currents,axis=-1,keepdims=True) # zero sum of currents.
        # currents *= 2
    else:
        currents = np.zeros((length,3))
        currents[:,0],currents[:,1],currents[:,2] = np.sin(rt),np.sin(rt-2./3.*pi),np.sin(rt-4./3.*pi)

        currents *= np.arange(length).reshape((length,1)).astype('float32')/length*2.0-1.0


    currents *= 1

    dq_fluxes = currents.copy()
    fluxes = dq_fluxes.copy()


    if model is None:
        for k in range(len(rt)):
            fluxes[k],dq_fluxes[k] = torque(currents[k],rt[k])
        return currents,fluxes,dq_fluxes,rt

    else:
        #if we have trained models:
        pass # foget bout this

    # print('---',currents.shape,dq_fluxes.shape)
    # print('generate',currents[:,0:20],dq_fluxes[:,0:20])



import matplotlib.pyplot as plt

split = 200
tot = split*1.5
ci,f,dqf,rp = generate_data_pair(tot)

print(rp.shape)

Xcurrent_input = ci[0:split,0:2]
Xrotor_phase = rp[0:split]
Ytorque = dqf[0:split,1]
print(Xcurrent_input.shape,Xrotor_phase.shape,Ytorque.shape)

Xcitest,Xrptest,Ytortest = ci[split:tot,0:2],rp[split:tot],dqf[split:tot,1]

psp=keras.layers.advanced_activations.ParametricSoftplus
elu = keras.layers.advanced_activations.ELU
def build_motor_model():
    rotor_phase = Input(shape=(1,) ,name='rotor_phase')
    current_input = Input(shape=(2,) ,name='current_input') # just phase a and b

    ext = Lambda(lambda x:K.concatenate([x,-x[:,0:1]-x[:,1:2]])) # extract third phase.

    # ci = ext(current_input)
    ci = current_input
    # ci = Dense(3,name='3p')(current_input)


    sinc = Lambda(lambda x:K.tanh(K.tanh(x)))

    def mul(x,y):
        return merge([x,y],mode='mul')

    def D(l):
        def imd(k,name=''):
            k = Dense(l,name=name)(k)
            k = Activation('tanh')(k)
            return k
        return imd

    def feat3(i):
        i=D(3)(i)
        i=D(5)(i)
        return i

    def quad(x):
        lsin = Lambda(lambda x:K.sin(x))
        lsin2 = Lambda(lambda x:K.sin(x+2*pi/3))
        lsin3 = Lambda(lambda x:K.sin(x+4*pi/3))

        lcos = Lambda(lambda x:K.cos(x))
        return merge([lsin(x),lcos(x)],mode='concat')

    cart = quad(rotor_phase)

    cart = feat3(cart)
    # i = feat3(current_input)
    i = feat3(ci)
    prod = mul(cart,i)

    torque = Dense(1,name='fin')(prod)

    model = Model(input=[rotor_phase,current_input],output=[torque])
    return model

motor_model = build_motor_model()
motor_model.summary()

def test():

    plt.ion()
    fig = plt.figure('test')
    figax = fig.add_subplot(111)

    testdata = generate_data_pair(500,random=False)
    ci,f,dqf,rp = testdata

    torq = motor_model.predict([rp,ci[:,0:2]])
    gt_torq = dqf[:,1:2]
    # print(torq.shape,gt_torq.shape)

    ae = np.abs(torq - gt_torq)
    maxe = np.max(ae)
    loss = np.mean(ae) # mae
    print('tested',ae.shape,'mae loss:',loss,'max torq e:',maxe)

    # plt.plot(r,np.hstack((currents,fluxes,dq_fluxes)))
    # plt.legend(['ia','ib','ic','fa','fb','fc','fq','fd','fz'])
    # # plt.plot(r,r)
    # plt.show()

    # print('ci',ci[0:10])

    # print(ci.shape,gt_torq.shape,torq.shape)
    stack = np.hstack((ci,f))
    stack2 = np.hstack((dqf[:,0:2],torq))
    # print(stack.shape)
    # print(stack[10])
    figax.clear()
    figax.plot(rp,stack,alpha=0.2)
    figax.plot(rp,stack2)
    figax.legend(['ia','ib','ic','fa','fb','fc','fd','gt_torq(fq)','pred_torq'],loc=3)
    # plt.plot(r,r)
    fig.canvas.draw()
    plt.pause(0.001)

def wrap(f):
    def testwrapper():
        if not hasattr(testwrapper,'timer'):
            testwrapper.timer = time.time()

        if testwrapper.timer - time.time()<-3:
            testwrapper.timer = time.time()
            f()

    return testwrapper

from histlogger import EpochEndCallback,LoggerCallback
logger = LoggerCallback(keys=[{'loss':[],'val_loss':[]}])
eec = EpochEndCallback(wrap(test))

def r(ep=10,opt=None,lr=0.01,loss='mse',bs=100):
    callbacks=[logger,eec]

    model = motor_model

    sgd = SGD(lr=lr, decay=1e-4, momentum=0.95, nesterov=True)
    # opt = Adam()
    if opt is None:
        opt = sgd
        callbacks.append(sgdr.gen_scheduler(maxlr=lr,t0=10,tm=1))

    model.compile(#loss='categorical_crossentropy',
                    loss=loss,
                    optimizer=opt,
                    # metrics=['accuracy'],
                    # metrics=['']
                    )

    model.fit([Xrotor_phase,Xcurrent_input],Ytorque,
            validation_data=([Xrptest,Xcitest],Ytortest),
              batch_size=bs,
              nb_epoch=ep,
              callbacks=callbacks)

def find_out():
    # build a model to find out optimal currents for a given torque and rotor angle

    # 1. generate samples
    # find out: given the torque and rotor angle, what current generate this torque most efficiently?

    c = 2000
    np.random.seed(42)
    rotor_angles = np.arange(c).reshape(c,1).astype('float32')/c*349.1389*pi
    # desired_torque = np.random.normal(loc=0.,scale=.5,size=(c,1)).astype('float32')

    desired_torque = np.arange(c).reshape(c,1).astype('float32')/c
    desired_torque = (desired_torque)**4 *.4

    null_input = Input(shape=(None,),name='null')
    angle_input = Input(shape=(1,),name='angle_input')

    from keras.regularizers import l1,l2

    #w/o reg .1492 loss 1.29 mean_weight_metric
    optimal_currents = MultipleInputLayer(
    output_dim=(c,2), name='optimal_currents', W_regularizer=l2(1e-6)
    )(null_input)

    torques = motor_model([angle_input,optimal_currents])

    newmodel = Model(input=[angle_input,null_input],output=torques)

    newmodel.summary()

    return rotor_angles,desired_torque,newmodel,c

ra,dt,nm,count_samples = find_out()

def rn(ep=10,opt=None,lr=0.01,loss='mse',bs=10000):

    model = nm

    # freeze all layers
    for l in motor_model.layers:
        l.trainable = False

    model.summary()

    sgd = SGD(lr=lr, decay=1e-4, momentum=0.95, nesterov=True)
    # opt = Adam()
    if opt is None:
        opt = sgd

    def mean_weight_metric(y_true, y_pred):
        return K.mean(K.abs(model.get_layer('optimal_currents').W))

    model.compile(#loss='categorical_crossentropy',
                    loss=loss,
                    optimizer=opt,
                    metrics=[mean_weight_metric],
                    # metrics=['']
                    )

    model.fit([ra,np.zeros((count_samples,))],dt,
              batch_size=count_samples,
              shuffle=False, # !!!!!!!!!important.
              nb_epoch=ep,
              callbacks=[rec])

    # unfreeze all layers
    for l in motor_model.layers:
        l.trainable = True

def show_weight():
    w = nm.get_layer('optimal_currents').get_weights()[0]
    print('optimal i', w[0:3])
    print('des torq',dt[0:3])
    print('rot ang',ra[0:3])

    predt = motor_model.predict([ra[0:3],w[0:3]])
    print('pred torq',predt)

def vis():
    plt.ion()
    w = nm.get_layer('optimal_currents').get_weights()[0]
    #(10000,3)

    actual_torque = motor_model.predict([ra,w])
    print(dt.shape,w.shape,actual_torque.shape)

    stack = np.hstack((dt,w,actual_torque))

    fig = plt.figure('find_out')
    figax = fig.add_subplot(111)

    figax.clear()
    figax.plot(ra,stack)

    # figax.plot(rp,stack,alpha=0.2)
    # figax.plot(rp,stack2)
    figax.legend(['des_torq','ia_opt','ib_opt','actual_torque'],loc=3)
    # plt.plot(r,r)
    fig.canvas.draw()
    plt.pause(0.001)

def vis2():
    plt.ion()
    c = 1000
    rotor_angles = np.arange(c).astype('float32')/c*10*pi
    # desired_torque = np.random.normal(loc=0.,scale=.8,size=(c,1)).astype('float32')

    desired_torque = np.arange(c).reshape(c,1).astype('float32')/c *.8
    desired_torque = (desired_torque)**2

    currents = rm.predict([rotor_angles,desired_torque])
    actual_torque = motor_model.predict([rotor_angles,currents])

    stack = np.hstack((desired_torque,currents,actual_torque))

    fig = plt.figure('rm visualizations')
    figax = fig.add_subplot(111)

    figax.clear()
    figax.plot(rotor_angles,stack)

    # figax.plot(rp,stack,alpha=0.2)
    # figax.plot(rp,stack2)
    figax.legend(['commanded_torq','ia_pred','ib_pred','actual_torque'],loc=3)
    # plt.plot(r,r)
    fig.canvas.draw()
    plt.pause(0.001)


rec = EpochEndCallback(wrap(vis))

def get_reverse_model():
    torq_input = Input(shape=(1,),name='torque_command')
    ang_input = Input(shape=(1,),name='angle_input')

    torq = torq_input

    def mul(x,y):
        return merge([x,y],mode='mul')

    def feat(i):
        i = Dense(3,activation='tanh')(i)
        # i = Dense(3,activation='tanh')(i)
        # i = Dense(3,activation='tanh')(i)
        i = Dense(12,activation='tanh')(i)
        # i = Dense(6,activation='tanh')(i)
        return i

    def unfeat(i):
        i = Dense(4,activation='tanh')(i)
        # i = Dense(4,activation='tanh')(i)
        i = Dense(4,activation='tanh')(i)
        i = Dense(2)(i)
        return i

    def quad(x):
        lsin = Lambda(lambda x:K.sin(x))
        lcos = Lambda(lambda x:K.cos(x))
        return merge([lsin(x),lcos(x)],mode='concat')

    cart = quad(ang_input)

    cart = feat(cart)
    torq = feat(torq)
    # torq = Dense(6,activation='tanh')(torq)

    prod = mul(cart,torq)

    # ia = unfeat(prod)
    # ib = unfeat(prod)
    # currents = merge([ia,ib],mode='concat')
    currents = unfeat(prod)

    rm = Model(input=[ang_input,torq_input],output=[currents])

    rm.summary()
    return rm
rm = get_reverse_model()

rec2 = EpochEndCallback(wrap(vis2))
def rmtrain(ep=10):
    w = nm.get_layer('optimal_currents').get_weights()[0]

    def myloss(y_true,y_pred):
        motor_model_func = K.function([motor_model.layers[0].input],
                                  [motor_model.layers[-1].output])

        loss = K.mean((motor_model_func(y_true)[0] - motor_model_func(y_pred)[0])**2,axis=-1)
        return loss
        # layer_output = get_3rd_layer_output([X])[0]

    rm.compile(loss='mse',optimizer='adam')
    rm.fit([ra,dt],w,
              batch_size=len(w),
              shuffle=False, # !!!!!!!!!important.
              nb_epoch=ep,
              callbacks=[rec2])
