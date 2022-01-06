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

    # apply binary thresholding
    contours, hierarchy = cv2.findContours(image=img_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    drawing = np.zeros(np.shape(img)[:-1])
    cv2.drawContours(image=drawing, contours=contours, contourIdx=-1, color=255, thickness=2, lineType=cv2.LINE_AA)

    thinned = cv2.ximgproc.thinning(drawing)


    cv2.imshow('Binary image', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Thined image', thinned)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
