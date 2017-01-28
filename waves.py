import pydub
import numpy as np
import numpy.fft as nf
import sounddevice as sd

fs = 44100

AudioSegment = pydub.AudioSegment

song = AudioSegment.from_mp3('lean_on.mp3')

entire = song[:].raw_data
npsong = np.fromstring(string=entire,dtype='int16')
npsong = npsong.reshape((len(npsong)/2,2))

def play(tensor):
    sd.play(tensor,fs)

def waterfall(tensor):
    #build a waterfall

    window = 512 # even number, 2^n best
    tiling = 3
    windows = len(tensor)/window * tiling - tiling

    wfall = np.zeros((windows,window/2+1),dtype='float32')

    for i in range(windows):
        start = i * window / tiling
        sliced = tensor[start:start+window,0]
        sp = spectrum(sliced)
        # print(sp.shape)
        wfall[i,:] = sp

    return wfall

def spectrum(tensor1d):
    coeffs = nf.rfft(tensor1d) # complex coeffs
    return np.absolute(coeffs) # magnitude

def show_waterfall(tensor):
    img = waterfall(tensor)
    # toning
    img = np.log(img + 5000) # add a small number, then log scale
    img -= np.min(img)
    img = img/np.max(img)
    # img = np.power(img,.5)
    # print(img.shape)
    # print(np.max(img),np.min(img))
    import vis
    vis.autoscaler_show(img.T,limit=800.)
