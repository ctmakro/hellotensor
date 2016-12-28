import tensorflow as tf
import numpy as np

a = tf.constant([[1.,2.,1.],[1.,2.,1.]])
sm = tf.nn.softmax(a)
sess = tf.Session()
print(sess.run(sm))
