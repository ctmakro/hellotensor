# chainer!

import chainer
import numpy as np

from chainer import *

import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

data = np.array([5],dtype='float32')
f = L.Linear(3,2) # dense MLP

x = Variable(data)

y = x**2 + F.sin(x)

def r(ep=10):
    for i in range(ep):
        print('ep',i)
        print('y_data:',y.data)
        y.run()
        y.backward()
        x.data -= x.grad
