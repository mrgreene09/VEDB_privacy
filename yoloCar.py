#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:01:41 2021

@author: peterriley
"""

import cv2 
import numpy as np
import blur_functions as bf
import plate_detector as aldp


whT = 320
confThreshold =0.5
nmsThreshold= 0.2




#name of image
name = "/Users/peterriley/Desktop/tv.png"


#### LOAD MODEL
## Coco Names
classesFile = "Yolo wieghts and names/coco.names"
classNames = []
with open(classesFile, "rt") as f:
    classNames=f.read().strip('\n').split('\n')
#print(classNames)
## Model Files
modelConfiguration = "Yolo wieghts and names/yolov3-tiny.cfg"
modelWeights = "Yolo wieghts and names/yolov3-tiny.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    plate_coords = []
    screen_coords = []
    for i in indices:
        
        i = i[0]
        if classIds[i] in [2, 3, 5, 7]: #classIds refering to things that might have a license plate
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
           # print(x,y,w,h)
            plate_coords.append(box)
              
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            #cv2.putText(img,f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
           
            
            
            
        elif classIds[i] in [62, 63, 67]: #different types of screens
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
           # print(x,y,w,h)
            screen_coords.append(box)
              
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            #cv2.putText(img,f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            
    return plate_coords, screen_coords

img = cv2.imread(name)
blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)
layersNames = net.getLayerNames()
outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(outputNames)
plate_coords, screen_coords = findObjects(outputs,img)

if len(plate_coords)>=1:
    for i in range(len(plate_coords)):
        magic = aldp.alpd(img, plate_coords[i][0], plate_coords[i][1], plate_coords[i][2], plate_coords[i][3]) 
else:
    magic = img

cv2.imshow("ImageZ", magic)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(4):
    cv2.waitKey(1)
    
#currently only returns screen coords and puts a box around them (no blurring yet)
