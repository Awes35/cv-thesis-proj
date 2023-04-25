#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:45:17 2023

@author: kollengruizenga
"""

import cv2

print (cv2.__version__)

proj_dir = "/Users/kollengruizenga/Desktop/Thesis Project/"

#%% load the input image and show its dimensions

image = cv2.imread(f"{proj_dir}/opencv-examples/clown.png")
(h, w, d) = image.shape
print('width={}, height={}, depth={}'.format(w, h, d))


#%% open with OpenCV and press a key on our keyboard to continue execution

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

