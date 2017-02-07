import cv2
import numpy as np

def gen_image_description():
    h,w = 128,128

    im = np.random.normal(loc=0.5,scale=0.3,size=(h,w)).astype('float32')
    blur = cv2.blur(im,(15,15))
    blur2 = cv2.blur(im,(7,7))
    im = im * .2 + blur * .5 + blur2 * .3
    # im = cv2.blur(im,(3,3))

    # print(im.shape)

    im[im>0.6] = 1.
    im[im<.4] = 0.

    # rows = np.random.choice(2)+1
    # cols = np.random.choice(3)+1
    rows,cols = 2,4
    description = ''

    hbias = 45 # float/drown
    rhb = np.random.rand()*hbias
    for r in range(rows):
        basey = (h-hbias)/rows*(r+0.5) + rhb

        wbias = 30
        whb = np.random.rand()*wbias
        for c in range(cols):
            basex = (w-wbias)/cols*(c+0.5) + whb

            char = np.random.choice(17)
            if char>=10:
                char = ''
            else:
                char = str(char)

            description += char

            scale = np.random.rand()*.8 + 1.

            brightness = np.random.choice(2)

            # correction
            textx = basex - 18 + np.random.normal(loc=0.,scale=1.)
            texty = basey + 15 + np.random.normal(loc=0.,scale=1.)

            place_text(im,text=char,
            coord=(int(textx),int(texty)),brightness=brightness,scale=scale,linewidth=2)

    im = cv2.blur(im,(5,5))
    im = cv2.resize(im,(32,32),interpolation=cv2.INTER_CUBIC)
    return im,description

def place_text(im,text,coord=(10,10),brightness=.5,scale=0.5,linewidth=1):

    # color = (255,255,255) # for color

    font = cv2.FONT_HERSHEY_SIMPLEX if np.random.rand()>0.5 else cv2.FONT_HERSHEY_COMPLEX

    color = (brightness,)
    cv2.putText(im, text,
		coord, font, scale, color, linewidth, cv2.LINE_AA)

def show_image(im):
    cv2.imshow('show',im)
    cv2.waitKey(1)

def loop_gen(count):
    imlist = []
    lblist=[]
    for i in range(count):
        im,desc = gen_image_description()
        # print(desc)
        # show_image(im)

        imlist.append(im)
        lblist.append(desc)
    return imlist,lblist

def generate_and_save(count):
    i,l = loop_gen(count)
    i = np.array(i)
    i = i*255.
    i = i.reshape(i.shape+(1,))
    i = i.astype('uint8')

    print(i.shape,i.dtype)

    l = np.array(l)
    print(l.shape,l.dtype)

    fname = 'variable_length_number_dataset.npz'

    with open(fname,'wb') as f:
        np.savez(f,images=i,labels=l)

    print('test...')
    with open(fname,'rb') as f:
        npz = np.load(f)
        print(len(npz['images']),len(npz['labels']))

    print('saved to '+fname)
