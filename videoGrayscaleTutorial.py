#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:48:42 2021

Showing how to write a grayscale movie in opencv

@author: michellegreene
"""

import cv2

# hard coding frames per second
fps = 30

# hard coding codec
fourcc = 'mp4v'

# establish input and output directories
videoPath = '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/cats.mp4'
outputPath = '/Volumes/etna/Scholarship/Michelle Greene/Students/Shared/GRAYcats.mp4' # note that this includes file name

# open a video object
vid = cv2.VideoCapture(videoPath)

# get video properties
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
videoSize = (width, height)
print('This video is {} by {} pixels'.format(width, height))

# get number of frames in video
frameCount = vid.get(cv2.CAP_PROP_FRAME_COUNT)
print('This video has: {} frames'.format(frameCount))

# create a video writer object
writer = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*fourcc), fps, videoSize)

# read in one frame at a time
count = 0
while vid.isOpened():
    success, image = vid.read()
    # assumes that any failure is the end of the video
    if not success:
        break
    
    # convert frame to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    count += 1
    print('Converted frame: {}'.format(count))
    
    # write changed frame to new video
    writer.write(image)
    
# release the input and output objects
vid.release()
writer.release()
