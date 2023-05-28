# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Open the h5 file containing the original images
f5 = 'D:/BAT/m-nattereri_20210924/Windtunnel_Prey_Capture_Zoom_1632474507_20210924_110827_S001_Mnat_2021_36_3.h5'
# Create title for plotting.  OBS only works for filepath above.
parts = f5.split('_')
NAME = '_'.join(parts[7:-1])

folder_path = "D:/BAT/m-nattereri_20210924"
output_folder = "D:/BAT/m-nattereri_20210924/sampling results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#%% sharpness
def sharpness(image):
	# compute the sobel mean score of the image and then return the sharpness
	# measure
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.abs(sobelx) + np.abs(sobely)
    return np.mean(sobel)
#%% blurriness
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

#%% camera 1 sharpness - OBS filepath for save is not accurate - should be moved to the sampling results folder

with h5py.File(f5, "r") as s:
    data = s['/Video/Camera_1/Mraw/Data']
    j_values = [3,5,10,15,20,30,40]
    sharpness_score = {}
    for j in j_values:
        sharpness_score[j] = []
    frame_count=data.shape[0]
    for j in j_values:
        for i in range(0, frame_count, j):
            img_h5 = data[i,:,:]
            img1_h5 = cv2.cvtColor(np.squeeze(img_h5),cv2.COLOR_GRAY2BGR)
            ss = sharpness(img1_h5)
            sharpness_score[j].append((i, ss))
                 

with open('9kernel_sampling_sharpness_score.json', 'w') as f:
    json.dump(sharpness_score, f)
#%% Bluriness camera 1 - OBS filepath for save is not accurate - should be moved to the sampling results folder
with h5py.File(f5, "r") as s:
    data = s['/Video/Camera_1/Mraw/Data']
    j_values = [3,5,10,15,20,30,40]
    bluriness_score = {}
    for j in j_values:
        bluriness_score[j] = []
    frame_count=data.shape[0]
    for j in j_values:
        for i in range(0, frame_count, j):
            img_h5 = data[i,:,:]
            img1_h5 = cv2.cvtColor(np.squeeze(img_h5),cv2.COLOR_GRAY2BGR)
            bs = variance_of_laplacian(img1_h5)
            bluriness_score[j].append((i, bs))
                 

with open('bluriness_sharpness_score.json', 'w') as f:
    json.dump(bluriness_score, f)
#%% Plotting the results - sharpness - OBS check that the filename is correct
with open('D:/BAT/m-nattereri_20210924/sampling results/sampling_sharpness_score.json', 'r') as f:
    sharpness_score = json.load(f)
j_values = [3,5,10,15,20,30,40]
# Convert the strings to integers
plot_j_values = [str(j) for j in j_values]
    
fig, axs = plt.subplots(len(j_values), figsize=(10,5*len(j_values)))

for i, j in enumerate(plot_j_values):
    frames = [t[0] for t in sharpness_score[j]]
    ss = [t[1] for t in sharpness_score[j]]
    axs[i].plot(frames, ss)
    axs[i].set_title(f"Sampling rate: {j}")
    axs[i].set_ylabel('Sobel score')
fig.suptitle('Effect of Different Sampling Rates on Sobel Score')
plt.tight_layout()
fig.savefig('D:/BAT/m-nattereri_20210924/sampling results/sampling_sobel_plots.png')
plt.show()

#%% Plotting the results - bluriness
with open('D:/BAT/m-nattereri_20210924/sampling results/bluriness_sharpness_score.json', 'r') as f:
    bluriness_score = json.load(f)

# Convert the strings to integers
plot_j_values = [str(j) for j in j_values]
    
fig, axs = plt.subplots(len(j_values), figsize=(20,3*len(j_values)))

for i, j in enumerate(plot_j_values):
    frames = [t[0] for t in bluriness_score[j]]
    ss = [t[1] for t in bluriness_score[j]]
    axs[i].plot(frames, ss)
    axs[i].set_title(f"Sampling Bluriness score for {j}")
    axs[i].set_ylabel('Laplacian variance score')

plt.tight_layout()
plt.show()

#%% Alternative sharpness sampling plot
import matplotlib.pyplot as plt
import numpy as np
import json

with open('D:/BAT/m-nattereri_20210924/sampling results/sampling_sharpness_score.json', 'r') as f:
    sharpness_score = json.load(f)
    
j_values = [3, 5, 10, 15, 20, 30, 40]
colors = plt.cm.tab10(np.linspace(0, 1, len(j_values)))

fig, ax = plt.subplots(figsize=(10, 5))

for j, color in zip(j_values, colors):
    frames = [t[0] for t in sharpness_score[str(j)]]
    ss = [t[1] for t in sharpness_score[str(j)]]
    ax.plot(frames, ss, color=color, label=f"Sampling rate: {j}")

ax.set_title('Effect of Different Sampling Rates on Sobel Score', fontsize = 20, fontweight='bold')
ax.set_xlabel('Frames', fontsize = 16, fontweight='bold')
ax.set_ylabel('Sobel score', fontsize = 16, fontweight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()

j_values = [3, 10, 15, 20]
colors = plt.cm.viridis(np.linspace(0, 1, len(j_values)))

fig, ax = plt.subplots(figsize=(10, 5))

for j, color in zip(j_values, colors):
    frames = [t[0] for t in sharpness_score[str(j)]]
    ss = [t[1] for t in sharpness_score[str(j)]]
    ax.plot(frames, ss, color=color, label=f"Sampling rate: {j}")

ax.set_title('Effect of Different Sampling Rates on Sobel Score', fontsize = 20, fontweight='bold')
ax.set_xlabel('Frames', fontsize = 16, fontweight='bold')
ax.set_ylabel('Sobel score', fontsize = 16, fontweight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend()
plt.tight_layout()
plt.show()
