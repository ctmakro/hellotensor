from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AtrousConvolution2D
from keras.optimizers import SGD,Adam,Adadelta,Nadam
from keras.utils import np_utils
import keras
from math import *
import keras.backend as K
import sgdr
import numpy as np

def torque(currents,theta):

    # currents are ampere
    # theta is radian
    # 4x3 pole

    fluxes = np.tanh(currents*4)/4
    # fluxes = currents

    # torque T == magnetic orientation cross rotor orientation

    def park(tensor,theta):
        park_matrix = np.array([
        [cos(theta),cos(theta-2/3*pi),cos(theta+2/3*pi)],
        [sin(theta),sin(theta-2/3*pi),sin(theta+2/3*pi)],
        [1/2,1/2,1/2]
        ])
        return np.dot(park_matrix,tensor.T).T

    dq_fluxes = park(fluxes,theta)

    return fluxes,dq_fluxes

def generate_data_pair(length):
    # r: electrical phase array
    # rt: rotor phase array
    # cs: currents array

    # r = np.random.rand(length)*2*pi
    rt = np.random.rand(length)*2*pi

    # currents = np.zeros(r.shape+(3,))
    # currents[:,0],currents[:,2],currents[:,1] = np.sin(r),np.sin(r+2/3*pi),np.sin(r+4/3*pi)

    currents = np.random.rand(length,3)*2-1
    currents[:,0:3] -= np.mean(currents,axis=-1).reshape((length,1))

    currents *= 1

    dq_fluxes = currents.copy()
    fluxes = dq_fluxes.copy()

    for k in range(len(r)):
        fluxes[k],dq_fluxes[k] = torque(currents[k],rt[k])

    print('---',currents.shape,dq_fluxes.shape)
    # print(currents[:,0:20],dq_fluxes[:,0:20])
    return currents,fluxes,dq_fluxes,rt

import matplotlib.pyplot as plt

r = np.arange(72)
currents,fluxes,dq_fluxes,rotor_phases = generate_data_pair(72)

plt.plot(r,np.hstack((currents,fluxes,dq_fluxes)))
plt.legend(['ia','ib','ic','fa','fb','fc','fq','fd','fz'])
# plt.plot(r,r)
plt.show()
