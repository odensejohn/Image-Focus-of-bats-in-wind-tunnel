# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:12:05 2023

@author: odens
"""

import pandas as pd
from deepposekit.io.video import VideoH5Reader
import numpy as np
import cv2


f5 = 'D:/BAT/Mix/Flight_Room_Prey_Capture_1666270056_20221020_144736_S002_Ppyg_2022_181_9.h5'
df = pd.read_json('D:/BAT/Mix/Yolo2_results/Flight_Room_Prey_Capture_1666270056_20221020_144736_S002_Ppyg_2022_181_9/Flight_Room_Prey_Capture_1666270056_20221020_144736_S002_Ppyg_2022_181_9_15frame_YOLO2_Data.json')

mask = df['Frame'] == 675
frame = df.loc[mask]

Image_data_1 = VideoH5Reader(f5, "Camera_2", batch_size=1, uint8=True, gray=True, imadjust='clahe')
img1 = cv2.cvtColor(np.squeeze(Image_data_1[675]), cv2.COLOR_GRAY2BGR)
i = frame['Frame'].iloc[0]
ss = frame['Camera_1_Face_sobel'].iloc[0]
x1 = int(frame['Camera_1_x1'].iloc[0])
#print(x1)
x2 = int(frame['Camera_1_x2'].iloc[0])
#print(x2)
y1 = int(frame['Camera_1_y1'].iloc[0])
#print(y1)
y2 = int(frame['Camera_1_y2'].iloc[0])
text = f'Frame: {i}  Sobel: {ss}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
text_x = img1.shape[1] - text_size[0] - 10
text_y = img1.shape[0] - text_size[1] - 10
cv2.putText(img1, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
cv2.imshow('Ppyg', img1)
cv2.imwrite('D:/BAT/Mix/Yolo2_results/Flight_Room_Prey_Capture_1666270056_20221020_144736_S002_Ppyg_2022_181_9/Ppyg_Frame_675.png', img1)
#%% Exploring inverted images.  Not worth continuing with
inverted_image = cv2.bitwise_not(img1)
cv2.imshow('inverted', inverted_image)
#%% Concatenate images

img1 = cv2.imread('D:/BAT/Mix/Yolo2_results/Flight_Room_Prey_Capture_1666270056_20221020_144736_S002_Ppyg_2022_181_9/Ppyg_Frame_675.png')
img2 = cv2.imread('D:/BAT/Mix/Yolo2_results/Windtunnel_Hand_Release_1659690063_20220805_110103_S005_Eser_2022_169_1/Eser_Frame_540.png')

concat_image = cv2.hconcat([img1, img2])
h, w = concat_image.shape[:2]

top = int(h*0.3)
right = int(w*0.8)
bottom = int(h*0.9)
left = int(w*0.2)

cropped_image = concat_image[top:bottom, left:right]

cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('D:/BAT/Mix/concat_bat_faces.png', cropped_image)
