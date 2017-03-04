import numpy as np
import scipy
import csv
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

def softmax(x):
    # x is array
    e_x = np.exp(x)
    return e_x / e_x.sum()

source_dir = '/Users/chia/DroneSamples'
#source_dir = 'D:/DroneSamples'

def loadcsv(path):

    print('trying to load csv from:',path)
    csvfile = open(path, newline='') # new to python3
    reader = csv.reader(csvfile)

    xarr = []
    yarr = []

    for line in reader:
        imgname = line[0]
        if imgname != 'filename' :
            xoff = int(line[1])
            yoff = int(line[2])
            scale = int(float(line[3]))

            exists = line[4] # 1 -> exist 0 -> n/e

            exists = int(exists)

            xarr.append(imgname)
            yarr.append({'xoff':xoff,'yoff':yoff,'scale':scale,'exists':exists})
            # yarr.append(exists)
            # yarr.append(exists*2-1)

    csvfile.close()
    print(len(xarr),'items loaded.')
    return xarr,yarr
#csv

def load_img_from_file(xarr):
    print('tring to load',len(xarr),'imgs from file')
    images = np.zeros((len(xarr), 64, 64, 1), dtype="uint8")
    for index,i in enumerate(xarr):
        imgdata = load_img(source_dir+'/composites/'+ i +'.png',grayscale=True,target_size=None)
        images[index,:,:,:] = np.reshape(imgdata,(64,64,1))
    return images

def load_one(path):
    imgdata = load_img(path,grayscale=True,target_size=None)
    imgdata = np.array(imgdata)
    print(imgdata.shape)
    tfstyle = np.reshape(imgdata,(imgdata.shape[0],imgdata.shape[1],1))
    return tfstyle

def load_data(wantfresh=False):
    xp = './ximages.npy'
    yp = './yvalues.npy'
    try:
        if wantfresh:
            raise('explicitly load from source.')
        ximages = np.load(xp)
        yvalues = np.load(yp)
        print('successfully loaded from cache')
    except:
        xarr,yarr = loadcsv(source_dir+'/composite_data.csv')

        # yvalues = np.reshape(yarr,(len(yarr),1)) # this is old code, yarr is an array of boolean 0 or 1

        # new code for yvalues in new architecture
        def limit(a):
            return (a if a<63 else 63) if a>0 else 0

        yvalues = np.zeros((len(yarr),4,4),dtype='float32')
        for index,y in enumerate(yarr):
            # construct heat image from description
            blank = np.zeros((64,64),dtype='uint8') #limitation
            xstart=limit(int(32+y['xoff']-y['scale']/2))
            xend = limit(int(32+y['xoff']+y['scale']/2))
            ystart=limit(int(32+y['yoff']-y['scale']/2))
            yend = limit(int(32+y['yoff']+y['scale']/2))

            blank[ystart:yend,xstart:xend] = 255 # paint it white

            blank = scipy.misc.imresize(blank,(4,4),interp='bilinear')
            yvalue = blank.astype('float32')/255

            # yvalue = softmax(yvalue)
            if index<10:
                print(index)
                print(blank)
                print(yvalue)
            yvalues[index] = yvalue

        ximages = load_img_from_file(xarr)
        np.save(xp,ximages)
        np.save(yp,yvalues)
        print('saved to cache.')

    #yvalues = np.reshape(yarr,(len(yarr),len(yarr[0])))
    print('ximages shape:',ximages.shape)
    print('yvalues shape:',yvalues.shape)

    return ximages,yvalues
