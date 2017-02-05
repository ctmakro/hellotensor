import tensorflow as tf

_tfsession_ = None
def get_session():
    global _tfsession_
    if _tfsession_ is None:
        _tfsession_ = tf.Session()
    return _tfsession_

def set_session(s):
    global _tfsession_
    _tfsession_ = s

def set_remote_session(ipad):
    set_session(tf.Session('grpc://'+ipad+':16384'))
