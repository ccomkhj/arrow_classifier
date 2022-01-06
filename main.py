import numpy as np
import cv2
from os import walk
import os
import matplotlib.pyplot as plt

path = 'template'
images = []
for (dirpath, dirnames, filenames) in walk(path):
    images.extend(filenames)
    break

for image in images:
    img = cv2.imread(os.path.join(path,image))

    # convert the image to grayscale format
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # leave only arrow
    ret,arrow = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)

    # skeltonization
    thinned = cv2.ximgproc.thinning(arrow.astype(np.uint8))

    contours, hierarchy = cv2.findContours(image=thinned, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # keep only the main contour (ignore noise)
    contour = max(contours, key=len)

    # simplify the contour using Douglas
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)

    drawing = np.zeros(np.shape(img)[:-1])
    cv2.drawContours(drawing,[approx],0,255,1)
    print('next')
    #TODO: check the point where the orientation changes, and compute the consine value between angles.

    cv2.imshow('skeleton arrow', drawing)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
