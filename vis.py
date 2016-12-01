import matplotlib.pyplot as plt
import math

def show(img):
    plt.imshow(img,cmap='gray')
    plt.show(block=False)

def getimage(index):
    xi = ximages[index]
    i = np.reshape(xi,(xi.shape[0],xi.shape[1]))
    return i

def showindex(index):
    show(getimage(index))

def showmatrix(m,w,h):
    fig0,plots = plt.subplots(h,w,subplot_kw={'xticks': [], 'yticks': []})
    fig0.subplots_adjust(hspace=0.01, wspace=0.01)
    index = 0
    for line in plots:
        for i in line:
            if index >= len(m):
                break
            i.imshow(m[index],cmap='gray')
            index+=1
    plt.show(block=False)
