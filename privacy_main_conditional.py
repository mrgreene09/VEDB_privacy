#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:27:50 2021

Modules: MTCNN face detection,
         YOLO (cars), to be implemented later
         YOLO (phones and screens) to be implemented later
Version 2: only detects faces if YOLO has detected a person

@author: michellegreene
@author: peterriley
"""

import cv2
#import skvideo.io
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN

# construct the argument parser
# parser = argparse.ArgumentParser(description='A script to blur faces and other sensitive material in video frames in a folder and save them in a new folder.')
# parser.add_argument("-i","--input", help="Path to input video. Make sure it only contains images.")
# parser.add_argument("-o","--output", help="Path to where the new video is to be saved.")
# args = vars(parser.parse_args())

# establish video reader and writer
# hard coding codec
fourcc = 'avc1'

# open a video object
#vid = cv2.VideoCapture(args["input"])
inName = 'losAngeles.mp4'
outName = 'newLA.mp4'
#inName = '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/cats.mp4'
#outName =  '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/cats2.mp4'
vid = cv2.VideoCapture(inName)

# get video properties
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
videoSize = (width, height)
fps = vid.get(cv2.CAP_PROP_FPS)

# define YOLO parameters
CONF_THRESH, NMS_THRESH = 0.1, 0.5
config = "./yolo_files/yolov3-tiny.cfg"
weights = "./yolo_files/yolov3-tiny.weights"

# Load the network
net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# create video writer
#outputFile = args["output"]
# writer = skvideo.io.FFmpegWriter(outputFile, outputdict={
#   '-vcodec': 'libx264',  #use the h.264 codec
#   '-crf': '0',           #set the constant rate factor to 0, which is lossless
#   '-preset':'veryslow'   #the slower the better compression, in princple, try 
#                          #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
# }) 

writer = cv2.VideoWriter(outName, cv2.VideoWriter_fourcc(*fourcc), fps, videoSize, True)

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
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(image)
    return  faces  

def yoloPerson(image):
    isPerson = False
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id==0 and confidence>CONF_THRESH: #person
                isPerson = True
                
    return isPerson

##### Main processing loop
count = 0
while vid.isOpened():
    
    # read a frame
    success, image = vid.read()
    # assumes that any failure is the end of the video
    if not success:
        break
    
    # counter for debugging
    count += 1
    if count%10==0:
        print('Processed: {} frames'.format(count))
    
    # check to see if there is a person in the image
    isPerson = yoloPerson(image)
    if isPerson:
    
        # initialize mask
        mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        
        # create blurred version of entire image
        scrambled = blur(image)
        
        # detect faces in image
        faceCoordinates = face_detect(image)
        
        # for each face, convert bounding box to ellipse
        for j in range(len(faceCoordinates)): #j = which face in the frame
            x,y, width, height = (faceCoordinates[j]['box'])
            #converts the bounding box to an ellipse via a custom function
            ellipse = rect_to_ellipse(x, y, width,  height)
            #puts the ellipse onto the mask
            cv2.ellipse(mask, ellipse[0], ellipse[1], 0, 0, 360, 255, -1)
            
         # apply logical masking to each face
        newImage = logical_mask(image, scrambled, mask)
    else:
        newImage = image
    
    # write newImage as a frame
    #writer.writeFrame(newImage[:,:,::-1])  #write the frame as RGB not BGR
    writer.write(newImage)
    
# release the input and output objects
vid.release()
writer.release()
#writer.close()