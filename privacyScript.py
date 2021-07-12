#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:01:40 2021

Privacy script

@author: peterriley
@author: michellegreene
"""
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import glob


input_path = "/*" #path to folder of images. Note, the path must end with /* to actually access the images
output_path  = "" #path to folder where the blurred images are to be saved

def rect_to_ellipse(x, y, width,  height):
    vert_axis  = round(height/2)
    horz_axis = round(width/2)
    center_x = round(x + horz_axis)
    center_y =  round(y +  vert_axis)
    center_coordinates = (center_x, center_y)
    axesLength = (horz_axis, vert_axis)    
    return center_coordinates, axesLength

# note: 15 is the local window - this is arbitrary and we may change it later
def blur(image):
    blurred = cv2.medianBlur(image, 15)
    return blurred

def face_detect(image):
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(image)
    return  faces    

def load_image(file):
    image = cv2.imread(file)
    #convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def logical_mask(image, scrambled, mask):
    fg = cv2.bitwise_or(scrambled, scrambled, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(image, image, mask=mask)
    newImage = cv2.bitwise_or(fg, bk)
    return newImage

# create list of images
imageList = glob.glob(input_path) # need to put pathway to images here

# main loop
for i in range(len(imageList)):
    # load image
    image = load_image(imageList[i])
    
    # detect faces in image
    faceCoordinates = face_detect(image)
    
    # create blurred version of entire image
    scrambled = blur(image)
    
    # create mask of zeros
    mask = np.full((scrambled.shape[0], scrambled.shape[1]), 0, dtype=np.uint8)
    
    # for each face, convert bounding box to ellipse
    for j in range(len(faceCoordinates)): #j = which face in the frame
        x,y, width, height = (faceCoordinates[j]['box'])
        #converts the bounding box to an ellipse via a custom function
        ellipse = rect_to_ellipse(x, y, width,  height)
        #puts the ellipse onto the mask
        cv2.ellipse(mask, ellipse[0], ellipse[1], 0, 0, 360, 255, -1)
    
    # apply logical masking to each face
    newImage = logical_mask(image, scrambled, mask)
    
    # write new image to disk
    filename = 'new'+imageList[i]
    cv2.imwrite(filename, newImage)
        

                      
                    

