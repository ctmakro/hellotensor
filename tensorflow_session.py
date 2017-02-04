import tensorflow as tf

_tfsession_ = tf.Session()
def get_session():
    return _tfsession_
