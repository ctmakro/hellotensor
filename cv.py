import numpy as np
import cv2
import math
import random

#cv2.startWindowThread()

def show(image,title='untitled'):
    cv2.namedWindow(title)
    cv2.imshow(title,image)
    return cv2.waitKey(2)

img = np.zeros((256,256,3),np.uint8)

def loopshow():
    j=0
    k=0
    l=0
    def ri(i=1.0):
        return (random.random()-0.3)*i
    def s(i):
        return math.sin(i)/2+0.5
    for i in range(256):
        for m in range(256):
            j+=ri(0.3)
            k+=ri(0.4)
            l+=ri(0.9)
            img[i,m,0] = s(j)*255
            img[i,m,1] = s(k)*255
            img[i,m,2] = s(l)*255
    show(img)

c = cv2.VideoCapture()

def cs():
    c.open(0)
    stat,im = c.read()
    c.release()
    show(im)
