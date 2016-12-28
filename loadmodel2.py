import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from loaddata import load_data,load_one
import math

# returns a compiled model
# identical to the previous one
model = load_model('evenbetter.h5')

ximages,yvalues = load_data()

def getimage(index):
    xi = ximages[index]
    i = np.reshape(xi,(xi.shape[0],xi.shape[1]))
    return i

def imshow(index):
    plt.figure()
    plt.imshow(getimage(index),cmap='gray')
    plt.show(block=False)

def imrange(start=0,count=2):
    import math
    s = count+5
    plt.figure()
    fig0, plots = plt.subplots(count,s,subplot_kw={'xticks': [], 'yticks': []})
    fig0.subplots_adjust(hspace=0.1, wspace=0.01)
    index=0
    for line in plots:
        for i in line:
            picnum = start+index
            p = predict(picnum)
            p = 1 if p[0,1]>p[0,0] else 0
            real = yvalues[picnum,0]
            if p==real:
                #nice guess
                i.imshow(getimage(picnum),cmap='gray')
            else:
                i.imshow(getimage(picnum),cmap='hot')
            i.set_title(str(picnum)+': '+str(p)+'('+str(real),fontsize=5)
            print(p,real)
            index+=1
    plt.show(block=False)

def weightshow(m):
    # m(w,h,d,count)
    import matplotlib.pyplot as plt
    import math
    #show a matrice of images
    count = m.shape[3]
    plt.figure()
    fig0, plots = plt.subplots(count,subplot_kw={'xticks': [], 'yticks': []})
    fig0.subplots_adjust(hspace=0.1, wspace=0.01)
    for i in range(count):
        img = np.reshape(m[:,:,:,i],(m.shape[0],m.shape[1]))
        plots[i].imshow(img,cmap='gray')
    plt.show(block=False)
def c1w_show():
    import vis
    c1w = model.layers[0].get_weights()[0]
    vis.weightshow(c1w)

def predict(index):
    xi = ximages[index]
    i = np.reshape(xi,(1,xi.shape[0],xi.shape[1],1))
    i = i.astype('float32')
    i/=255
    i-=0.5
    return model.predict(i)

def false_positives():
    i = ximages.astype('float32')
    i/=255
    i-=0.5
    print ('start prediction..')
    res = model.predict(ximages)
    print ('done...')
    fp=0
    fn=0
    for index,r in enumerate(res):
        if r[1]>r[0] and yvalues[index]==0:
            fp+=1
            print('false positive on',index,r[1])
        if r[0]>r[1] and yvalues[index]==1:
            fn+=1
    print('total fp:',fp,'total fn:',fn)

def get_output(X):
    import keras.backend as K
    import matplotlib.pyplot as plt

    targetlayer = model.layers[-1] #last

    try: # compile only once!
        get_output.func
    except:
        get_output.func = K.function([model.layers[0].input,K.learning_phase()], [targetlayer.output])

    res = get_output.func([X,0])
    imd = res[0]

    return imd

def get_output_of(index):
    img = ximages[index]
    img = img.astype('float32')
    inp = img/255-0.5
    return get_output([inp])

def get_dense_weight():
    dense = model.layers[15]
    dw = dense.get_weights()
    return dw[0],dw[1]

def softmax(x):
    # x is array
    e_x = np.exp(x)
    return e_x / e_x.sum()

def plot_rel(start=0,count=2):
    c = count*(count+5)
    images=[]
    for i in range(c):
        i33 = get_output_of(i+start)
        gt33 = yvalues[i+start]
        # print(i33)
        img = ximages[i+start].astype('float32')
        newimg = np.zeros([img.shape[0],img.shape[1],3],dtype='float32')
        newimg[:,:,0:3]=img[:,:]/255

        for row in range(3):
            for col in range(3):
                kx = 14+row*12
                ky = 14+col*12
                newimg[kx:kx+12,ky:ky+12,0]+= i33[0][row][col][0]/50
                newimg[kx:kx+12,ky:ky+12,1]+= gt33[row][col]/5
                newimg = np.clip(newimg,0,1.0)
        images.append(newimg)

    import vis
    vis.showmatrix(images,count+5,count)

def testit():
    import vis
    img = load_one('test.jpg')
    w,b = get_dense_weight()
    res = np.reshape(img, (img.shape[0],img.shape[1]))

    fi = img.astype('float32')
    fi/=255
    fi-=0.5

    z = 64

    for i in range(0,img.shape[0]-z,z/2):
        for j in range(0,img.shape[1]-z,z/2):
            zone = fi[i:i+z,j:j+z,:]
            pool = get_output([zone])[0][0][0]
            manual_perdiction = softmax(np.add(np.dot(pool,w),b))
            print(manual_perdiction)
            print(i,j,'scanned',manual_perdiction[1])
            if manual_perdiction[1]>0.5:
                b = 2
                c = z-b
                res[i:i+z,j:j+z]+=12
                # res[i+b:i+c,j+b] += 20
                # res[i+b:i+c,j+c] += 20
                # res[i+b,j+b:j+c] += 20
                # res[i+c,j+b:j+c] += 20
    vis.show(res)
