import numpy as np
import random
import sys
import canton as ct
from canton import *
import tensorflow as tf

time_steps = 16

def get_text_data(filename):
    import codecs
    with open(filename,'rb') as f:
        text = f.read()
    length = len(text)
    print('got corpus length:', length)
    return text

def model_builder():
    c = ct.Can()
    gru,d1,d2 = (
        c.add(GRU(256,256)),
        c.add(LastDimDense(256,64)),
        c.add(LastDimDense(64,256)),
    )

    def call(i,starting_state=None):
        # i is one-hot encoded
        i = gru(i,starting_state=starting_state)
        # (batch, time_steps, 512)
        shape = tf.shape(i)
        b,t,d = shape[0],shape[1],shape[2]

        ending_state = i[:,t-1,:]

        i = d1(i)
        i = Act('elu')(i)
        i = d2(i)
        i = Act('softmax')(i)

        return i, ending_state
    c.set_function(call)
    return c

def feed_gen():
    input_text = tf.placeholder(tf.uint8,
        shape=[None, None]) # [batch, timesteps]

    input_text_float = tf.one_hot(input_text,depth=256,dtype=tf.float32)

    xhead = input_text_float[:,:-1] # [batch, 0:timesteps-1, 256]
    gt = input_text_float[:,1:] # [batch, 1:timesteps, 256]
    y,_ = model(xhead,starting_state=None) # [batch, 1:timesteps, 256]

    def cross_entropy_loss_per_char(pred,gt): # last dim is one_hot
        def log2(i):
            return tf.log(i) * 1.442695
        return - tf.reduce_sum(log2(pred+1e-14) * gt, axis=tf.rank(pred)-1)

    loss = ct.cross_entropy_loss(y,gt)
    loss_per_char = cross_entropy_loss_per_char(y,gt)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(
        loss,var_list=model.get_weights())

    def feed(minibatch):
        nonlocal train_step,loss,input_text
        sess = ct.get_session()
        res = sess.run([loss,train_step],feed_dict={input_text:minibatch})
        return res[0]

    # stateful predict:
    # if we have starting_state for the RNN
    starting_state = tf.placeholder(tf.float32, shape=[None, None])
    stateful_y, ending_state = \
        model(input_text_float,starting_state=starting_state)

    # if we dont have starting state for the RNN
    stateful_y_init, ending_state_init = \
        model(input_text_float)

    def stateful_predict(st,i):
        sess = ct.get_session()
        if st is None: # if swe dont have starting_state for the RNN
            res = sess.run([stateful_y_init,ending_state_init],
                feed_dict={input_text:i})
        else:
            res = sess.run([stateful_y,ending_state],
                feed_dict={input_text:i,starting_state:st})
        return res

    def loss_statistics(i):
        sess = ct.get_session()
        res = sess.run([loss_per_char],
            feed_dict={input_text:i})
        return res

    return feed, stateful_predict, loss_statistics

# if you are using IPython:
# run r(1000) to train the model
# run show2(1000) to generate text

def r(ep=100):
    length = len(corpus)
    batch_size = 256
    mbl = time_steps * batch_size
    sr = length - mbl - time_steps - 2
    for i in range(ep):
        print('---------------------iter',i,'/',ep)

        j = np.random.choice(sr)

        minibatch = corpus[j:j+mbl]
        minibatch.shape = [batch_size, time_steps]

        loss = feed(minibatch)
        print('loss:',loss)

        if i%100==0 : pass#show2()

class utf8statemachine:
    # byte(int)s in. character out (when ready). raises error when illegal.
    def __init__(self):
        self.state=0
        self.buffer = []
    def flush(self):
        char = str(bytes(self.buffer),'utf-8')
        self.buffer = []
        return char
    def bytein(self,b):
        # assume b is uint.
        if self.state==0: # normal mode
            if b & 0b10000000 == 0:# if first bit is 0
                self.buffer.append(b)
                return self.flush()
            if b & 0b11110000 == 0b11100000: # if starts with 1110
                self.state=2
                self.buffer.append(b)
                return None
            if b & 0b11100000 == 0b11000000: # if starts with 110
                self.state=1
                self.buffer.append(b)
                return None
            raise NameError('byte should start with 0b0 or 0b110 or 0b1110')
        
        if self.state>0:
            if b & 0b11000000 == 0b10000000:# if starts with 10
                self.state -=1
                self.buffer.append(b)
            else:
                raise NameError('byte should start with 0b10')
            
            if self.state==0:
                return self.flush()
            return None       
            
def show2(length=400):
    import sys,os
    asc_buf = np.fromstring('\n',dtype='uint8').reshape(1,1)
    starting_state = None
    sm = utf8statemachine()
    errors = 0
    # sequentially generate text out of the GRU
    for i in range(length):
        stateful_y, ending_state = predict(starting_state,asc_buf)

        dist = stateful_y[0,0] # last dimension is the probability distribution
        code = np.random.choice(256, p=dist) # sample a byte from distribution

        try:
            result = sm.bytein(code) # put in utf8 state machine
            if result is not None: # if the state machine spit out a character
                sys.stdout.write(result) # write to stdout
                
            # accept the result if no utf-8 decoding error detected
            asc_buf[0,0] = code
            starting_state = ending_state
        except NameError: # if decoding error
            # sys.stdout.write('e')
            errors += 1
            # don't accept the results, try sample again next iteration
        
        if i%10==0:
            sys.stdout.flush()
            pass
       
    sys.stdout.flush()
    print('')
    print('total UTF-8 decoding errors:',errors)

# bullshit analyzer
def bsa(text):
    buf = np.fromstring(text,dtype='uint8').reshape(1,len(text))
    # what is the entropy? start from 2nd byte
    loss, = loss_stats(buf)

    simplified = text[0]

    print(text[0],'initial')
    for i in range(1,len(text)):
        print(text[i],loss[0,i-1])
        if loss[0,i-1] < 1: # discard words that are less than 1-bit
            simplified+='-'
        else:
            simplified+=text[i]

    print('simplified:',simplified)

argv = sys.argv
 
if len(argv)<2:
    print('(Error)please provide a filename as the first argument. The file should be in UTF-8 encoding, without BOM.')
else:
    text = get_text_data(argv[1]) # the string
    corpus = np.fromstring(text,dtype='uint8') # the bytes
    print('corpus loaded. corpus[0]:',corpus[0], 'text[0]:',text[0])

    model = model_builder()

    feed, predict, loss_stats = feed_gen()

    sess = ct.get_session()
    sess.run(tf.global_variables_initializer())
