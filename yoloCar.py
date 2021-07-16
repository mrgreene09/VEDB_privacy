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
name = "please.jpg"


#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, "rt") as f:
    classNames=f.read().strip('\n').split('\n')
#print(classNames)
## Model Files
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
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
    
    for i in indices:
        
        i = i[0]
        if classIds[i] in [2, 3, 5, 7]: #classIds refering to things that might have a license plate
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # print(x,y,w,h)
            #img = aldp.alpd(img, x, y, w, h)    #this line search for the license plate within the rectangle and blur!  
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            #cv2.putText(img,f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
       

img = cv2.imread(name)
blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)
layersNames = net.getLayerNames()
outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(outputNames)
findObjects(outputs,img)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(4):
    cv2.waitKey(1)