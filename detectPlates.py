#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:01:15 2021

@author: michellegreene
"""

import cv2
import imutils
import numpy as np

# read input image
img = cv2.imread('/path/to/image')

# convert to grayscale and apply slight blurring to make edge detection easier
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 15, 15) 

# use canny filter to find edges
edged = cv2.Canny(gray, 30, 200)

# grab the 10 longest contours 
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# check that the contours form a rectangle and is a reasonable aspect ratio
for c in contours:
    
    # rectangle checking
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    if len(approx) == 4:
        screenCnt = approx
        
        # now let's check the aspect ratio: standard US plates are 2x wide as long
        (w, h) = cv2.boundingRect(c)[2:]
        aspectRatio = w / float(h)
        
        if (aspectRatio > 1.5 and aspectRatio < 3): 
            break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

# use elementwise masking
mask = np.zeros(gray.shape, np.uint8)
newImage = cv2.bitwise_and(img, img, mask=mask)

# get bounding box coordinates
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
