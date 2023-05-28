# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:57:45 2023

@author: odens
"""

from ultralytics import YOLO
#%% Model training

model = YOLO('yolov8n.pt')

model.train(data='D:/BAT/YOLO2/data/data.yaml', epochs=20, imgsz=1024, workers = 1)

#%%
model = YOLO('D:/BAT/YOLO2/runs/detect/train3/weights/best.pt')
results = model.val(data='D:/BAT/YOLO2/data/data.yaml', split = 'test', batch = 1, imgsz=1024, workers=1)


    