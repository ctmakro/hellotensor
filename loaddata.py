import numpy as np
import csv

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

def loadcsv(path):

    print('trying to load csv from:',path)
    csvfile = file(path, 'rb')
    reader = csv.reader(csvfile)

    xarr = []
    yarr = []

    for line in reader:
        imgname = line[0]

        xoff = line[1]
        yoff = line[2]
        scale = line[3]

        exists = line[4] # 1 -> exist 0 -> n/e

        if imgname != 'filename' :
            exists = int(exists)

            xarr.append(imgname)
            # yarr.append([xoff,yoff,scale])
            yarr.append(exists)
            # yarr.append(exists*2-1)

    csvfile.close()
    print(len(xarr),'items loaded.')
    return xarr,yarr
#csv

def load_img_from_file(xarr):
    print('tring to load',len(xarr),'imgs from file')
    images = np.zeros((len(xarr), 64, 64, 1), dtype="uint8")
    for index,i in enumerate(xarr):
        imgdata = load_img('/Users/chia/DroneSamples/composites/'+ i +'.jpg',grayscale=True,target_size=None)
        images[index,:,:,:] = np.reshape(imgdata,(64,64,1))
    return images

def load_one(path):
    imgdata = load_img(path,grayscale=True,target_size=None)
    imgdata = np.array(imgdata)
    print(imgdata.shape)
    tfstyle = np.reshape(imgdata,(imgdata.shape[0],imgdata.shape[1],1))
    return tfstyle

def load_data():
    xp = './ximages.npy'
    yp = './yvalues.npy'
    try:
        ximages = np.load(xp)
        yvalues = np.load(yp)
        print('successfully loaded from cache')
    except:
        xarr,yarr = loadcsv('/Users/chia/DroneSamples/composite_data.csv')
        ximages = load_img_from_file(xarr)
        yvalues = np.reshape(yarr,(len(yarr),1))

        np.save(xp,ximages)
        np.save(yp,yvalues)
        print('saved to cache.')

    #yvalues = np.reshape(yarr,(len(yarr),len(yarr[0])))
    print('ximages shape:',ximages.shape)
    print('yvalues shape:',yvalues.shape)

    return ximages,yvalues
