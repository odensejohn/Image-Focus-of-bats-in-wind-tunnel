# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:29:20 2023

@author: odens
"""

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from deepposekit.io.video import VideoH5Reader
from sklearn.preprocessing import MinMaxScaler

# Open the h5 file containing the original images
f5 = 'D:/BAT/m-nattereri_20210924/Windtunnel_Prey_Capture_Zoom_1632474507_20210924_110827_S003_Mnat_2021_36_3.h5'
# Create title for plotting.  OBS only works for filepath above.
# Extract the filename from the path
filename = os.path.basename(f5)

# Split the filename at the last occurrence of "/"
head, _, tail = filename.rpartition('/')

# Remove the extension ".h5" from the tail
NAME = tail.rpartition('.')[0]

folder_path = "D:/BAT/m-nattereri_20210924"
output_folder = "D:/BAT/m-nattereri_20210924/sampling results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
def sharpness(image, kernel_size):
	# compute the sobel mean score of the image and then return the sharpness
	# measure
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel = np.abs(sobelx) + np.abs(sobely)
    return np.mean(sobel)

Image_data_1 = VideoH5Reader(f5, "Camera_1", batch_size=1, uint8=True, gray=True)
frame_count = len(Image_data_1)
kernels = [3,5,7]
kernel_results = []
for i in range(frame_count):
    # Importing camera 1 image and adding data to image
    img1 = cv2.cvtColor(np.squeeze(Image_data_1[i]), cv2.COLOR_GRAY2BGR)
    item = {}
    for k in kernels:
        out = sharpness(img1, k)
        item[f'{k}_kernel'] = out
    item['Frame'] = i
    kernel_results.append(item)
    
df = pd.DataFrame(kernel_results)
scaler = MinMaxScaler()
df['Normalised_3'] = scaler.fit_transform(df[['3_kernel']])
df['Normalised_5'] = scaler.fit_transform(df[['5_kernel']])
df['Normalised_7'] = scaler.fit_transform(df[['7_kernel']])


fig, (ax, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True)

ax.plot(df['5_kernel'])
ax.set_title('5')
ax2.plot(df['7_kernel'])
ax2.set_title('7')

plt.show()

fig, ax = plt.subplots(figsize = (20,20))
fig.suptitle('Normalised Mean Sobel Scores for Differing Kernel Sizes', fontsize=24)
ax.plot(df['Normalised_3'], color='red', label = '3 Kernels')
ax.plot(df['Normalised_5'], color='blue', label = '5 Kernels')
ax.plot(df['Normalised_7'], color='orange', label = '7 Kernels')
ax.legend(title_fontsize=20, fontsize=16, loc='upper left')
ax.set_xlabel('Frame', fontsize=16)
ax.set_ylabel('Normalised Sobel Score', fontsize=16)
plt.savefig(f'{output_folder}/{NAME}_normalised_kernel_comparison.png')
plt.show()

#%% Visualisation of an image:
def sobel(image, kernel_size):
	# compute the sobel mean score of the image and then return the sharpness
	# measure
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel = np.abs(sobelx) + np.abs(sobely)
    return sobel 

def laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F)

img1 =   cv2.cvtColor(np.squeeze(Image_data_1[700]), cv2.COLOR_GRAY2BGR)  

kernel_size = 5  # Choose the appropriate kernel size
sharpness_score = sobel(img1, kernel_size)
laplacian_score = laplacian(img1)
sharpness_score_3 = sobel(img1, 3)
sharpness_score_7 = sobel(img1, 7)

# Obtain the binary edge map
threshold = 300  # Adjust this threshold as needed
threshold_3 =25
threshold_7 = 3000
threshold_laplacian = 20
binary_edges = np.where(sharpness_score > threshold, 255, 0).astype(np.uint8)
binary_edges_3 = np.where(sharpness_score_3 > threshold_3, 255, 0).astype(np.uint8)
binary_edges_7 = np.where(sharpness_score_7 > threshold_7, 255, 0).astype(np.uint8)
laplacian_edges = np.where(laplacian_score > threshold_laplacian, 255, 0).astype(np.uint8)

# Plotting the results
plt.figure(figsize=(20, 20))
plt.suptitle('Visualisation of Gradient Magnitude Per Pixel for Differing Kernel Sizes',
             fontsize = 20, fontweight='bold')
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original Image', fontsize = 16, fontweight='bold' )
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sharpness_score_3, cmap='gray')
plt.title('Kernel 3x3', fontsize = 16, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sharpness_score, cmap='gray')
plt.title('kernel 5x5', fontsize = 16, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sharpness_score_7, cmap='gray')
plt.title('kernel 7x7', fontsize = 16, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'{output_folder}/no threshold.png')
plt.show()

#df.to_json(f'{output_folder}/{NAME}_kernel_testing.json', orient= 'records')

#%% Batface results
# Open the h5 file containing the original images
f5 = 'D:/BAT/Mix/Windtunnel_Prey_Capture_Zoom_1632746366_20210927_143926_S001_Mnat_2021_36_4.h5'
# Create title for plotting.  OBS only works for filepath above.
# Extract the filename from the path
filename = os.path.basename(f5)

# Split the filename at the last occurrence of "/"
head, _, tail = filename.rpartition('/')

# Remove the extension ".h5" from the tail
NAME = tail.rpartition('.')[0]

folder_path = "D:/BAT/m-nattereri_20210924"
output_folder = "D:/BAT/m-nattereri_20210924/sampling results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
Image_data_1 = VideoH5Reader(f5, "Camera_1", batch_size=1, uint8=True, gray=True)

img1 =   cv2.cvtColor(np.squeeze(Image_data_1[510]), cv2.COLOR_GRAY2BGR) 

plt.imshow(img1, cmap='gray')
df = pd.read_json('D:/BAT/Mix/Yolo2_results/Windtunnel_Prey_Capture_Zoom_1632746366_20210927_143926_S001_Mnat_2021_36_4/Windtunnel_Prey_Capture_Zoom_1632746366_20210927_143926_S001_Mnat_2021_36_4_15frame_YOLO2_Data.json')
row = df[df['Frame'] == 510]
x1 = int(row['Camera_1_x1'].iloc[0])
#print(x1)
x2 = int(row['Camera_1_x2'].iloc[0])
#print(x2)
y1 = int(row['Camera_1_y1'].iloc[0])
#print(y1)
y2 = int(row['Camera_1_y2'].iloc[0])
#print(y2) 
face = img1[y1:y2, x1:x2,:]
plt.imshow(face, cmap='gray')

kernel_size = 5  # Choose the appropriate kernel size
sharpness_score = sobel(face, kernel_size)
laplacian_score = laplacian(face)
sharpness_score_3 = sobel(face, 3)
sharpness_score_7 = sobel(face, 7)


threshold = 300  # Adjust this threshold as needed
threshold_3 =21.5
threshold_7 = 3500
threshold_laplacian = 10
binary_edges = np.where(sharpness_score > threshold, 255, 0).astype(np.uint8)
binary_edges_3 = np.where(sharpness_score_3 > threshold_3, 255, 0).astype(np.uint8)
binary_edges_7 = np.where(sharpness_score_7 > threshold_7, 255, 0).astype(np.uint8)
laplacian_edges = np.where(laplacian_score > threshold_laplacian, 255, 0).astype(np.uint8)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(face, cmap='gray')
plt.title('image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_edges, cmap='jet')
plt.title('Sobel Edges')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_edges_3, cmap='jet')
plt.title('kernel 3')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_edges_7, cmap='jet')
plt.title('kernel 7 Edges')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Normalising and comparing thresholds
def normalise(array):
    norm_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return norm_array
# normalising results
sharpness_score = sobel(face, 5)
laplacian_score = laplacian(face)
sharpness_score_3 = sobel(face, 3)
sharpness_score_7 = sobel(face, 7)

sharpness_score = normalise(sharpness_score)
sharpness_score_3 = normalise(sharpness_score_3)
sharpness_score_7 = normalise(sharpness_score_7)

threshold = 0.15

binary_edges = np.where(sharpness_score > threshold, 255, 0).astype(np.uint8)
binary_edges_3 = np.where(sharpness_score_3 > threshold, 255, 0).astype(np.uint8)
binary_edges_7 = np.where(sharpness_score_7 > threshold, 255, 0).astype(np.uint8)
laplacian_edges = np.where(laplacian_score > threshold, 255, 0).astype(np.uint8)


# Plotting the results
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(face, cmap='gray')
plt.title('image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_edges, cmap='jet')
plt.title('Sobel Edges')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_edges_3, cmap='jet')
plt.title('kernel 3')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_edges_7, cmap='jet')
plt.title('kernel 7 Edges')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Exploring combined results  - https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/

def sobel_combined_image(image, kernel_size):
    # Calculate x and y scores
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Calculate weighted combined scores into single image
    combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return combined

img1 =   cv2.cvtColor(np.squeeze(Image_data_1[700]), cv2.COLOR_GRAY2BGR)


threshold_5 = 60  # Adjust this threshold as needed
threshold_3 =2
threshold_7 = 350

ss3 = sobel_combined_image(img1, 3)
ss5 = sobel_combined_image(img1, 5)
ss7 = sobel_combined_image(img1, 7)

binary_edges_3 = np.where(ss3 > threshold_3, 255, 0).astype(np.uint8)
binary_edges_5 = np.where(ss5 > threshold_5, 255, 0).astype(np.uint8)
binary_edges_7 = np.where(ss7 > threshold_7, 255, 0).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_edges_3, cmap='gray')
plt.title('Kernel 3')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_edges_5, cmap='gray')
plt.title('kernel 5')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_edges_7, cmap='gray')
plt.title('kernel 7')
plt.axis('off')

plt.tight_layout()
plt.show()
#%% Face images alone
face3 = sobel_combined_image(face, 3)
face5 = sobel_combined_image(face, 5)
face7 = sobel_combined_image(face, 7)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(face, cmap='gray')
plt.title('image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(face3, cmap='jet')
plt.title('Kernel 3')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(face5, cmap='jet')
plt.title('kernel 5')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(face7, cmap='jet')
plt.title('kernel 7')
plt.axis('off')

plt.tight_layout()
plt.show()
