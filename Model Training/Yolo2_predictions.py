# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:23:55 2023
Requirements for deepposekit: tensorflow, scikit-learn, imgaug
Requirements for gpu pytorch, command line install: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
then install ultralytics.  If no gpu required/available skip previous step.

@author: odens
"""
import h5py
import cv2
import numpy as np
from ultralytics import YOLO
from deepposekit.io.video import VideoH5Reader
import torch

# Import model
model = YOLO('D:/BAT/YOLO2/runs/detect/train3/weights/best.pt')

# Open the h5 file containing the original images
f5 = 'D:/BAT/p-pipistrellus_20221020/Flight_Room_Prey_Capture_1666256488_20221020_110128_S001_Ppip_2022_238_2.h5'

Image_data_1 = VideoH5Reader(f5, "Camera_1", batch_size=1, uint8=True, gray=True)
Image_data_2 = VideoH5Reader(f5, "Camera_2", batch_size=1, uint8=True, gray=True)
#%% Single image
img1 = cv2.cvtColor(np.squeeze(Image_data_2[700]), cv2.COLOR_GRAY2BGR)
#print(type(img1))
results = model.predict(img1)
results_visualise = results[0].plot()
cv2.imshow("result", results_visualise)
cv2.waitKey(0)
cv2.destroyAllWindows() 
#%% manipulating image with bounding box data

boxes = results[0].boxes

box = boxes.cpu().numpy()
coords = box.xyxy
#print(box.xyxy)
x1, y1, x2, y2 = coords[0].astype(int)
face = img1[y1:y2, x1:x2,:]
cv2.imshow("face", face)
cv2.waitKey(0)
#%% Whole dataset
               
for i in range(0,70):
    img1 = cv2.cvtColor(np.squeeze(Image_data_1[i]), cv2.COLOR_GRAY2BGR)
    result = model.predict(img1)
    if len(result[0].boxes) > 0:            
        boxes = result[0].boxes
        box = boxes.cpu().numpy()
        coords = box.xyxy  
        x1, y1, x2, y2 = coords[0].astype(int)
        face = img1[y1:y2, x1:x2,:]
        cv2.imshow("face", face)
        cv2.waitKey(0)

    else:
        pass
    
cv2.destroyAllWindows()    