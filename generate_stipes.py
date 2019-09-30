import numpy as np
import cv2

width  = 1000 
height = 1000

nstripes = 50
alpha = np.pi/2 # phase shift by 90 degree

def genFringe(h,w):

    fringe1 = np.zeros((h,w))
    fringe2 = np.zeros((h,w))
    fringe3 = np.zeros((h,w))
    fringe4 = np.zeros((h,w))
    # period length:
    phi = w/nstripes
    
    # scale factor delta = 2pi/phi
    delta = 2*np.pi/phi

    f = lambda x,a : (np.cos(x*delta+a) + 1) * 120
    # compute a row of fringe pattern
    sinrow1 = [f(x,0) for x in xrange(w)]
    sinrow2 = [f(x,alpha) for x in xrange(w)]
    sinrow3 = [f(x,2*alpha) for x in xrange(w)]
    sinrow4 = [f(x,3*alpha) for x in xrange(w)]

    fringe1[:,:] = sinrow1
    fringe2[:,:] = sinrow2
    fringe3[:,:] = sinrow3
    fringe4[:,:] = sinrow4

    return fringe1, fringe2, fringe3, fringe4

def genFringeHor():
    imgarr = genFringe(width,height)
    return map(rotate, imgarr)

def rotate(img):
    timg = img
    timg = cv2.transpose(img)
    cv2.flip(timg, 1)
    return timg

images = genFringeHor()
i=0
for img in images:
    img = img.astype(np.uint8)
    cv2.imshow('fringe', img)
    filename = 'fringe_{0}.png'.format(i)
    cv2.imwrite(filename, img)
    i = i+1
    cv2.waitKey(3)
