#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:01:15 2021
@author: michellegreene
Major editd on Wed Jul 14 14:38:18 2021
@author: peterriley
"""


import cv2
import imutils
import numpy as np
import blur_functions as bf



def alpd(img, xx, yy, ww, hh):
    roi = img[yy:yy+hh , xx:xx+ww] #extracts just the car rectangle to be analzed
# =============================================================================
#     cv2.imshow("Image", roi)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     for i in range(4):
#         cv2.waitKey(1)
# =============================================================================

    # convert to grayscale and apply slight blurring to make edge detection easier
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
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
        img[yy:yy+hh , xx:xx+ww] = roi
        return img
    
    
    else:
         detected = 1
         
         # use elementwise masking
         mask = np.zeros(gray.shape, np.uint8)
         newImage = cv2.drawContours(mask,[screenCnt],0,255,-1,)
         newImage = cv2.bitwise_and(roi, roi, mask=mask)
        
         # get bounding box coordinates
         (x, y) = np.where(mask == 255)
         (topx, topy) = (np.min(x), np.min(y))
         (bottomx, bottomy) = (np.max(x), np.max(y))
         
         #draws a rectangle on the image (useful for seeing where the function is actually blurring)
         roi = cv2.rectangle(roi,( topy, topx), (bottomy, bottomx), (0,255,0),10  )
         
         
         #blurs the identified rectangle
         roi = bf.blur_regions(roi, ( topy, topx), (bottomy, bottomx)) 
         
         img[yy:yy+hh , xx:xx+ww] = roi #adds the blurred part back onto the original image
        

         return img #importantly, this function overwrites the frame/image that does in.
     
        

# =============================================================================
# img =cv2.imread("/Users/peterriley/Desktop/cars.jpg")
# cv2.imshow("First", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for i in range(4):
#     cv2.waitKey(1)
# 
# 
# 
# #dog = alpd(img, 734 ,258, 521, 450)   #please.jpg
# #dog = alpd(img, 1295 ,985, 1667, 1260)  #cali
# dog = alpd(img, 91, 515, 920, 900)
# cv2.imshow("Image", dog)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for i in range(4):
#     cv2.waitKey(1)
# 
# =============================================================================

     
     
     
     
     
     
     
     
     
     
     
     
     
