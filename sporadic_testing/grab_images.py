#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 20:20:04 2023

@author: kollengruizenga
"""


import cv2 as cv
import numpy as np


#%% 

cam_capture = cv.VideoCapture(0)
if not cam_capture.isOpened():
    print("Cannot open capture feed.")
    exit()


ret, frame = cam_capture.read()

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

cv.imshow('frame', gray)


