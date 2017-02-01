import pydub
import numpy as np
import numpy.fft as nf
import sounddevice as sd
import vis

fs = 44100 # assume 44100Hz sampling rate

AudioSegment = pydub.AudioSegment

def loadfile(fname,type='mp3'):
    song = AudioSegment.from_file(fname,type)
    raw = song[:].raw_data
    npsong = np.fromstring(string=raw,dtype='int16') # assume int16
    npsong = npsong.reshape((len(npsong)/2,2)) # assume stereo
    return npsong

def play(tensor): # (length,2) or (length,) tensor
    sd.play(tensor,fs)

def waterfall(tensor1d):
    #build a waterfall

    window = 512 # even number, 2^n best
    tiling = 2 # overlapping between windows, larger -> more overlapping
    windows = len(tensor1d)/window * tiling - tiling

    wfall = np.zeros((windows,window/2+1),dtype='float32')

    hanning_window = np.hanning(window)
    # thx to https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/

    for i in range(windows):
        start = i * window / tiling
        sliced = tensor1d[start:start+window]
        sp = spectrum(sliced*hanning_window)
        # print(sp.shape)
        wfall[i,:] = sp

    return wfall

def spectrum(tensor1d):
    coeffs = nf.rfft(tensor1d) # complex coeffs
    return np.absolute(coeffs) # magnitude

def show_waterfall(tensor1d): # show waterfall of mono audio
    if tensor1d.shape[-1] == 2:
        print('this',tensor1d.shape,'is not mono audio')
        return False

    img = waterfall(tensor1d)
    # toning
    # img = np.log(img + 1000) # add a small number, then log scale

    img = np.power(img,0.4545*0.5) # 2.2 Gamma * 2 Gamma
    img -= np.min(img)
    img = img/np.max(img)

    vis.autoscaler_show(img.T,limit=1000.)

song = loadfile('lean_on.mp3',type='mp3')
