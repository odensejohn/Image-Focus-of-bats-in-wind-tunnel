# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:59:38 2023
Requirements for deepposekit: tensorflow, scikit-learn, imgaug
Requirements for gpu pytorch, command line install: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
then install ultralytics.  If no gpu required/available skip previous step.
@author: odensejohn
"""

import numpy as np
import cv2
import os
from ultralytics import YOLO
from deepposekit.io.video import VideoH5Reader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import time

'''
 Adjust the following variables depending on usage requirements:
 sampling_rate should be set to 1 if the whole video wants processing
 video can be True or False.  Adjust depending on whether the video is wanted.  
 The video takes a long time to generate if using a small sampling rate.
 normalised and sobel thresholds were optimum at 0.74 and 215 respectively 
 but can be adjusted as required here.
'''
sampling_rate = 1
video = False # change to False if video not wanted
normalised_threshold = 0.74
sobel_threshold = 215

# Load the bat face detection model

model = YOLO('D:/BAT/YOLO2/runs/detect/train3/weights/best.pt')

start = time.perf_counter()

folder_path = 'D:/BAT/pipistrellus'

def map_normalised_values(x, threshold):
    if x['Normalised_Camera_1_Face_sobel'] and x['Normalised_Camera_2_Face_sobel'] > threshold:
        return 1
    else:
        return 0
    
def map_sobel_values(x, threshold):
    if x['Camera_1_Face_sobel'] and x['Camera_2_Face_sobel'] > threshold:
        return 1
    else:
        return 0

#sobel
def sobel(image):
	# compute the sobel mean score of the image and then return the sobel score
	
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.abs(sobelx) + np.abs(sobely)
    return np.mean(sobel) 

for file_name in os.listdir(folder_path):
    if file_name.endswith('.h5'):
        # Import file   
        f5 = f'{folder_path}/{file_name}'
        # Converting to uint 8
        Image_data_1 = VideoH5Reader(f5, "Camera_1", batch_size=1, uint8=True, gray=True, imadjust='clahe')
        Image_data_2 = VideoH5Reader(f5, "Camera_2", batch_size=1, uint8=True, gray=True, imadjust='clahe')
        # Calculating frame count
        frame_count = len(Image_data_1)

        
        # Create title for plotting.  OBS only works for filepath above.
        # Extract the filename from the path
        filename = os.path.basename(f5)
        
        # Split the filename at the last occurrence of "/"
        head, _, tail = filename.rpartition('/')
        
        # Remove the extension ".h5" from the tail
        NAME = tail.rpartition('.')[0]
        
        # Create json folder for results if necessary
        json_folder = f"{folder_path}/Yolo2_results"
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
        # Create folder for results
        output_folder = f'{json_folder}/{NAME}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Calculate bounding box values where present. 
        
        camera1_json = []
        camera2_json = []
        complete_sobel_laplacian_json = [] 
        
        for i in range(0, frame_count, sampling_rate):
             
            img1 = cv2.cvtColor(np.squeeze(Image_data_1[i]), cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(np.squeeze(Image_data_2[i]), cv2.COLOR_GRAY2BGR)
            ss = round(sobel(img1), 2)
            ss2 = round(sobel(img2), 2)
            item = {'Frame' : i, 'Camera_1_sobel' : ss,
                    'Camera_2_sobel' : ss2}
            complete_sobel_laplacian_json.append(item)
            result_1 = model.predict(img1)
            result_2 = model.predict(img2)
            if len(result_1[0].boxes) > 0:            
                boxes = result_1[0].boxes
                boxconf = boxes.conf
                box = boxes.cpu().numpy()
                coords = box.xyxy  
                x1, y1, x2, y2 = coords[0].astype(int)
                face = img1[y1:y2, x1:x2,:]
                ss = round(sobel(face),2)
                item = {"Frame": i,
                        "Camera_1_Face_sobel": ss,
                        'Camera_1_x1':x1,
                        'Camera_1_y1':y1,
                        'Camera_1_x2':x2,
                        'Camera_1_y2':y2,
                        'Camera_1_BB Confidence': boxconf}
                # append to list
                camera1_json.append(item) 
            else:
                item = {"Frame": i,
                        "Camera_1_Face_sobel": "NA",
                        'Camera_1_x1': "NA",
                        'Camera_1_y1': "NA",
                        'Camera_1_x2': "NA",
                        'Camera_1_y2': "NA",
                        'Camera_1_BB Confidence' : "NA"}    
                camera1_json.append(item)
            
            if len(result_2[0].boxes) > 0:            
                boxes2 = result_2[0].boxes
                box2conf = boxes2.conf
                box2 = boxes2.cpu().numpy()
                coords = box2.xyxy  
                x1, y1, x2, y2 = coords[0].astype(int)
                face = img1[y1:y2, x1:x2,:]
                ss = round(sobel(face),2)
                item = {"Frame": i,
                        "Camera_2_Face_sobel": ss,
                        'Camera_2_x1':x1,
                        'Camera_2_y1':y1,
                        'Camera_2_x2':x2,
                        'Camera_2_y2':y2,
                        'Camera_2_BB Confidence' : box2conf}
                # append to list
                camera2_json.append(item) 
                    
            else:
                item = {"Frame": i,
                        "Camera_2_Face_sobel": "NA",
                        'Camera_2_x1': "NA",
                        'Camera_2_y1': "NA",
                        'Camera_2_x2': "NA",
                        'Camera_2_y2': "NA",
                        'Camera_2_BB Confidence' : "NA"}
                camera2_json.append(item)
        
        
        # Create and merge pandas dataframes
        
        df1 = pd.DataFrame(complete_sobel_laplacian_json)       
        df2 = pd.DataFrame(camera1_json)
        df3 = pd.DataFrame(camera2_json)
        
        result = pd.merge(df1, df2, on='Frame')
        result = pd.merge(result, df3, on='Frame')
        # print(result.columns)
        # Change values to numeric
        # loop through the columns of the dataframe
        for column in result.columns:
            # check if the column is of object type
            if result[column].dtype == 'object':
                # convert the column to numeric with coerce
                result[column] = pd.to_numeric(result[column], errors='coerce')
        
        face1max = result['Camera_1_Face_sobel'].max()
        face1min = result['Camera_1_Face_sobel'].min()
        result['Normalised_Camera_1_Face_sobel'] = (result['Camera_1_Face_sobel']-face1min)/(face1max-face1min)
        print(result['Normalised_Camera_1_Face_sobel'].head())
        print(result['Normalised_Camera_1_Face_sobel'].max())
        print(result['Normalised_Camera_1_Face_sobel'].min())
        
        face2max = result['Camera_2_Face_sobel'].max()
        face2min = result['Camera_2_Face_sobel'].min()
        result['Normalised_Camera_2_Face_sobel'] = (result['Camera_2_Face_sobel']-face2min)/(face2max-face2min)
        print(result['Normalised_Camera_2_Face_sobel'].head())
        print(result['Normalised_Camera_2_Face_sobel'].max())
        print(result['Normalised_Camera_2_Face_sobel'].min())
        
        # Adjust Threshold values here if necessary
        result['Normalised Threshold'] = result.apply(lambda x: map_normalised_values(x, normalised_threshold), axis=1)
        result['Sobel Threshold'] = result.apply(lambda x: map_sobel_values(x, sobel_threshold), axis=1)

        # Plot Predictions
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,20), sharex=True)
        fig.suptitle(f'Focus Model Predictions \n {NAME}')
        ax1.plot(result['Frame'], result['Normalised Threshold'])
        ax1.set_title('Normalised Sobel Prediction')
        ax1.set_ylabel('')
        ax1.set_yticks([0, 1])
        
        ax2.plot(result['Frame'], result['Sobel Threshold'])
        ax2.set_title('Sobel Prediction')
        ax2.set_ylabel('')
        ax2.set_yticks([0, 1])
        fig.savefig(f'{output_folder}/{NAME}_predictions.png')
        if video:
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out1 = cv2.VideoWriter(f'{output_folder}/{NAME}_y2_cam1.avi', fourcc, 20.0, (1024,1024))
            out2 = cv2.VideoWriter(f'{output_folder}/{NAME}_y2_cam2.avi', fourcc, 20.0, (1024,1024))
                
            
            for i in range(0, frame_count, sampling_rate):
                # Importing camera 1 image and adding data to image
                img1 = cv2.cvtColor(np.squeeze(Image_data_1[i]), cv2.COLOR_GRAY2BGR)
                row = result[result['Frame'] == i]
                if any(pd.isna(row.loc[row.index[0],['Camera_1_x1', 'Camera_1_x2', 'Camera_1_y1', 'Camera_1_y2']])):
                    out1.write(img1)                
                else:
                    ss = round(row['Normalised_Camera_1_Face_sobel'].iloc[0], 2)
                    x1 = int(row['Camera_1_x1'].iloc[0])
                    #print(x1)
                    x2 = int(row['Camera_1_x2'].iloc[0])
                    #print(x2)
                    y1 = int(row['Camera_1_y1'].iloc[0])
                    #print(y1)
                    y2 = int(row['Camera_1_y2'].iloc[0])
                    #print(y2)
                    text = f'Frame: {i}  Normalised Sobel: {ss}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x = img1.shape[1] - text_size[0] - 10
                    text_y = img1.shape[0] - text_size[1] - 10
                    cv2.putText(img1, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
                    out1.write(img1)
                
                # Importing camera 2 image and adding data to image
                img2 = cv2.cvtColor(np.squeeze(Image_data_2[i]), cv2.COLOR_GRAY2BGR)
                if any(pd.isna(row.loc[row.index[0],['Camera_2_x1', 'Camera_2_x2', 'Camera_2_y1', 'Camera_2_y2']])):
                    out2.write(img2)     
                else:
                    ss = round(row['Normalised_Camera_2_Face_sobel'].iloc[0],2)
                    x1 = int(row['Camera_2_x1'].iloc[0])
                    #print(x1)
                    x2 = int(row['Camera_2_x2'].iloc[0])
                    #print(x2)
                    y1 = int(row['Camera_2_y1'].iloc[0])
                    #print(y1)
                    y2 = int(row['Camera_2_y2'].iloc[0])
                    #print(y2)
                    text = f'Frame: {i}  Normalised Sobel: {ss}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x = img2.shape[1] - text_size[0] - 10
                    text_y = img2.shape[0] - text_size[1] - 10
                    cv2.putText(img2, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
                    out2.write(img2)
            out1.release()
            out2.release()
        else:
            pass
        
        # Export to JSON       
        result.to_json(f'{output_folder}/{NAME}_complete_Data.json', orient='records')
        # Remove Unwanted images
        # Create mask with desired threshold
        mask = (result['Camera_1_Face_sobel'] > 100) & (result['Camera_2_Face_sobel'] > 100)
        output = result.loc[mask]
        output.to_json(f'{output_folder}/{NAME}_batface_Data.json', orient='records')
        
    

end = time.perf_counter()
print(f'Time taken in seconds: {end - start}')

