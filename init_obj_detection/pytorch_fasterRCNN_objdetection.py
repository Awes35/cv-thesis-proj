#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:06:12 2023

@author: kollengruizenga
"""

import time

import torch # -- PyTorch
import cv2 # -- OpenCV

#import torchvision
from torchvision import models as torch_models
from torchvision import transforms as T

from PIL import Image #Pillow -- default image backend supported by TorchVision

import matplotlib.pyplot as plt


#%% FasterRCNN - ResNet50 backbone

model = torch_models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
#(pretrained=True) 
model.eval()


#Define class names - re: PyTorch docs
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



#%% Functions

def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
       
    """
    starttime = time.time()

    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    endtime = time.time()
    pred_time = endtime - starttime

    return pred_boxes, pred_class, pred_time



def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3): # Get predictions 
    boxes, pred_cls, pred_time = get_prediction(img_path, threshold) 
    print(f"Identified {len(boxes)} objects..")
    print(f"Predictions took {pred_time} seconds!")

    # Read image with cv2 
    img = cv2.imread(img_path) 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    starttime = time.time()
    # Convert to RGB 
    for i in range(len(boxes)): #Convert coordinate points to tuple of int's
        print(f"Object number {i+1} -- {pred_cls[i]}")
        
        pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
        
        # Draw Rectangle with the coordinates 
        cv2.rectangle(img,
                      pt1, # -- top left corner of rectangle
                      pt2, # -- bottom right corner of rectangle
                      color=(0, 255, 0), # -- line color of GREEN
                      thickness=rect_th # -- thickness of line
                      )
        
        # Write the prediction class 
        cv2.putText(img,
                    pred_cls[i],
                    pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size,
                    (0,255,0),
                    thickness=text_th
                    )
        
        # display the output image 
        plt.figure(figsize=(20,30)) 
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        #plt.show()
    
    endtime = time.time()
    plot_time = endtime - starttime
    print(f"Plotting boxes took {plot_time} seconds!")

    plt.show()
    #end


#%% Identify objects in pics

#wget https://www.wsha.org/wp-content/uploads/banner-diverse-group-of-people-2.jpg -O people.jpg

#object_detection_api('people.jpg')

object_detection_api('girl_cars.jpg', threshold=0.25, rect_th=2)

#object_detection_api('guydude.jpeg', rect_th=2, text_th=1, text_size=1)

#object_detection_api('flowerbug.jpeg', rect_th=2, text_th=1, text_size=1)

#object_detection_api('artgirl.jpeg', threshold=0.35, rect_th=2, text_th=1, text_size=1)

#object_detection_api('soldiers.jpeg', threshold=0.70, rect_th=1, text_th=1, text_size=0.75)



