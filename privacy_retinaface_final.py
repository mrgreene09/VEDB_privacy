#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:17:31 2021

Face privacy script. Detects faces via retinaface on 50% downsampled frames

@author: michellegreene
@author: peterriley
@author: abemieses
"""

import cv2
import numpy as np
from retinaface import RetinaFace
import pandas as pd
import skvideo.io

# read in the sessions file
sessions = pd.read_csv('/path/to/sessionList.csv')
sessionList = list(sessions['Session'])

scale_percentage = 50

# Helper functions
def rect_to_ellipse(x, y, width,  height):
    vert_axis  = round(height/2)
    horz_axis = round(width/2)
    center_x = round(x + horz_axis)
    center_y =  round(y +  vert_axis)
    center_coordinates = (center_x, center_y)
    axesLength = (horz_axis, vert_axis)    
    return center_coordinates, axesLength

def blur(image):
    blurred = cv2.medianBlur(image, 149) #149 pixels (must be odd number)
    return blurred

def logical_mask(image, scrambled, mask):
    fg = cv2.bitwise_or(scrambled, scrambled, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(image, image, mask=mask)
    newImage = cv2.bitwise_or(fg, bk)
    return newImage

def face_detect(image):
    #detector = MTCNN()
    # detect faces in the image
    #faces = detector.detect_faces(image)
    faces = RetinaFace.detect_faces(image)
    return  faces  

def resize(input_image, scale_percentage):
    width = int(input_image.shape[1] * scale_percentage / 100)
    height = int(input_image.shape[0] * scale_percentage / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    return resized

for i in range(len(sessionList)):
    inName = '/media/data/vedb/staging/'+sessionList[i]+'/world.mp4'
    outName = '/media/data/vedb/staging/'+sessionList[i]+'/worldPrivate.mp4'

    # open a video object
    vid = cv2.VideoCapture(inName)
    
    # get video properties
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoSize = (width, height)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    # create FFMPEG writer object
    writer = skvideo.io.FFmpegWriter(outName, outputdict={
      '-vcodec': 'libx264',  #use the h.264 codec
      '-crf': '18',           #set the constant rate factor to 0, which is lossless
      '-preset':'medium'   #the slower the better compression, in princple, try 
                              #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 
    
    
    ##### Main processing loop
    count = 0
    while vid.isOpened():
        
        # read a frame
        success, image = vid.read()
        # assumes that any failure is the end of the video
        if not success:
            break
        
        # show progress
        count += 1
        if count%100==0:
            percentDone = int(count/frameCount)*100
            print('Processed: {} percent'.format(percentDone))
            
        # initialize mask
        mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        
        # resize the original image
        image_resized = resize(image, scale_percentage)
        
        # create blurred version of entire image
        scrambled = blur(image)
    
        # detect faces in image
        faceCoordinates = face_detect(image_resized)
        
        # dictionary is returned if at least one face is detected
        if type(faceCoordinates) is dict:
    
            # for each face, convert bounding box to ellipse
            for j in range(len(faceCoordinates)):  # j = which face in the frame
                # get face coordinates from bounding box
                xmin, ymin, xmax, ymax = faceCoordinates['face_'+str(j+1)]['facial_area']
                width = xmax-xmin
                height = ymax-ymin
    
                # rescaled x, y, w, h
                x_rescaled = xmin / (scale_percentage / 100)
                y_rescaled = ymin / (scale_percentage / 100)
                width_rescaled = width / (scale_percentage / 100)
                height_rescaled = height / (scale_percentage / 100)
    
                # converts the bounding box to an ellipse via a custom function
                ellipse = rect_to_ellipse(x_rescaled, y_rescaled, width_rescaled, height_rescaled)
                
                # puts the ellipse onto the mask
                cv2.ellipse(mask, ellipse[0], ellipse[1], 0, 0, 360, 255, -1)
    
            # apply logical masking to each face
            newImage = logical_mask(image, scrambled, mask)
        else:
            newImage = image
            
        # write newImage as a frame
        writer.writeFrame(newImage[:,:,::-1])  #write the frame as RGB not BGR
        #writer.write(newImage)
        
    # release the input and output objects
    vid.release()
    writer.close()

