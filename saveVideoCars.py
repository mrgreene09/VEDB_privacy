#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:11:29 2021

Get images with cars for labeling license plates.

@author: michellegreene
"""

import cv2
import glob
import numpy as np

# path to video images
inputPath = '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/Lewiston/'

# path to output images
outputPath = '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/carImages/'

# make list of videos
vidList = sorted(glob.glob(inputPath+'*.mp4'))

# define YOLO parameters
CONF_THRESH, NMS_THRESH = 0.5, 0.5
config = "yolo_files/yolov3-tiny.cfg"
weights = "yolo_files/yolov3-tiny.weights"

# Load the network
net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# loop through videos
car = 0
for i in range(len(vidList)):
    # open video
    vid = cv2.VideoCapture(vidList[i])
    
    # loop through frames
    while vid.isOpened():
    
        # read a frame
        success, image = vid.read()
        # assumes that any failure is the end of the video
        if not success:
            break
        height, width = image.shape[:2]
        
        # use YOLOv3 to detect a car, bus, or truck
        # convert the image to blob and perform forward pass to get the 
        #bounding boxes with their confidence scores
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)
        
        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

            if confidence > CONF_THRESH:
                if class_id in [2, 5, 7]:
                    car += 1
                    # write the image to file
                    outname = 'car'+str(car)+'.jpg'
                    cv2.imwrite(outputPath+outname, image)
        
        
