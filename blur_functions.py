#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:20:49 2021

@author: peterriley
"""
import cv2
import numpy as np



def blur(image):
    blurred = cv2.medianBlur(image, 15)
    return blurred


def blur_regions(image, p1, p2):
    scrambled = blur(image)   
    background = image
    mask = np.full((scrambled.shape[0], scrambled.shape[1]), 0, dtype=np.uint8) 
    cv2.rectangle(mask, p1, p2, (255,255,255), -1) 
    fg = cv2.bitwise_or(scrambled, scrambled, mask=mask)
    mask = cv2.bitwise_not(mask)  
    bk = cv2.bitwise_or(background, background, mask=mask)
    frame = cv2.bitwise_or(fg, bk) #important to note, this is overwriting the array that held the original video (obviously not overwriting the original file)
    return frame
