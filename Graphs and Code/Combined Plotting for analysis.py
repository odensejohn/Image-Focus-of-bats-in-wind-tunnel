# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:26:40 2023

@author: odens
"""

import matplotlib.pyplot as plt
import os
import pandas as pd

folder_path = 'D:/BAT/Gold'
predictions_path = 'D:/BAT/Gold/Yolo2_results'
gold_path = 'D:/BAT/Gold/Standard'

for file_name in os.listdir(folder_path):
    if file_name.endswith('.h5'):
    # Import file   
        f5 = f'{folder_path}/{file_name}'
        
        # Create title for plotting.  OBS only works for filepath above.
        # Extract the filename from the path
        filename = os.path.basename(f5)
        
        # Split the filename at the last occurrence of "/"
        head, _, tail = filename.rpartition('/')
        
        # Remove the extension ".h5" from the tail
        NAME = tail.rpartition('.')[0]

        df_predictions = pd.read_json(f'{predictions_path}/{NAME}/{NAME}_best_predictions.json')
        df_gold = pd.read_json(f'{gold_path}/{NAME}_Gold.json')
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize = (15,20), nrows=4, sharex=True)
        fig.suptitle(f'Predictions for \n{NAME}', fontsize = 20, fontweight='bold')

        #ax1.plot(df_predictions['Combined 0.7'], label = 'Threshold = 0.7')
        #ax1.plot(df_predictions['Combined 0.75'], label = 'Threshold = 0.75')
        ax1.plot(df_predictions['Combined 0.74'], label = 'Sobel predictions with \nnormalised threshold of 0.74')
        ax1.legend(loc='upper left')
        ax1.set_ylabel('')
        ax1.set_yticks([0, 1])
        ax1.tick_params(axis='y', labelsize=14)

        #ax2.plot(df_predictions['Sharpness 150'], label = 'Threshold 150')
        #ax2.plot(df_predictions['Sharpness 175'], label = 'Threshold 175')
        ax2.plot(df_predictions['Sharpness 215'], label = 'Sobel predictions \nwith threshold 215')
        ax2.legend(loc='upper left')
        ax2.set_ylabel('')
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='y', labelsize=14)

        ax3.plot(df_predictions['frame_selection'], label = 'Deeppose kit \nunclustered predictions')
        ax3.plot(df_predictions['cluster_selection'], label = 'Deeppose kit \nclustered predictions')
        ax3.legend(loc='upper left')
        ax3.set_ylabel('')
        ax3.set_yticks([0, 1])
        ax3.tick_params(axis='y', labelsize=14)

        ax4.plot(df_gold['frame_idx'], df_gold['Gold'], label = 'Gold Standard')
        ax4.set_ylabel('')
        ax4.set_yticks([0, 1])
        ax4.legend(loc='upper left')
        ax4.tick_params(axis='x', labelsize=14)
        ax4.tick_params(axis='y', labelsize=14)
        fig.savefig(f'{predictions_path}/{NAME}/{NAME}_best_74_predictions_vs_gold.png')
        plt.show()

#%% Plotting for camera specific results     
import matplotlib.pyplot as plt
import os
import pandas as pd

def map_normalised_values(x, threshold):
    if x['Normalised_Camera_2_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0
    
def map_sharpness_values(x, threshold):
    if x['Camera_2_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0
    
df_predictions = pd.read_json('D:/BAT/Gold/Yolo2_results/Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4/Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4_best_predictions.json')
df_gold = pd.read_csv('D:/BAT/Gold/Standard/Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4_camera_2_focus_combined.csv')
df_gold['combined'] = df_gold['combined'].apply(lambda x: 0 if x !=2 else 1)

df = df_predictions[['Frame', 'Camera_2_Face_Sharpness', 'Normalised_Camera_2_Face_Sharpness']].copy()

# Calculate normalised threshold values
df['0.74'] = df.apply(lambda x: map_normalised_values(x, 0.74), axis = 1)

# Calculate sharpness thresholds

df['Sharpness 215'] = df.apply(lambda x: map_sharpness_values(x, 215), axis = 1)


fig, (ax1, ax2, ax4) = plt.subplots(figsize = (15,20), nrows=3, sharex=True)
fig.suptitle('Predictions for camera 2\nWindtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3', fontsize = 20, fontweight='bold')

ax1.plot(df['0.74'], label = 'Sobel predictions with \nnormalised threshold of 0.74', color = 'green')
ax1.legend(loc='upper left')
ax1.set_ylabel('')
ax1.set_yticks([0, 1], ['Out of Focus', 'In Focus'])
ax1.tick_params(axis='y', labelsize=14)
ax1.set_xlim(400, 1100)

ax2.plot(df['Sharpness 215'], label = 'Sobel predictions \nwith threshold 215', color = 'red')
ax2.legend(loc='upper left')
ax2.set_ylabel('')
ax2.set_yticks([0, 1], ['Out of Focus', 'In Focus'])
ax2.tick_params(axis='y', labelsize=14)
ax2.set_xlim(400, 1100)

ax4.plot(df_gold['frame_idx'], df_gold['combined'], label = 'Gold Standard', color = 'blue')
ax4.set_ylabel('')
ax4.set_xlabel('Frame', fontsize=16, fontweight='bold')
ax4.set_yticks([0, 1], ['Out of Focus', 'In Focus'])
ax4.legend(loc='upper left')
ax4.tick_params(axis='x', labelsize=14)
ax4.tick_params(axis='y', labelsize=14)
ax4.set_xlim(400, 1100)
fig.savefig('D:/BAT/Gold/Standard/Combined_images/Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4_camera2.png')
plt.show()

#%% One plot visualisation.

# Create a figure and an axis object
fig, ax = plt.subplots(figsize=(15, 20))

# Plot each line on the same axis object
ax.plot(df['Frame'], df['0.74'], color='red', label='Sobel predictions with \nnormalised threshold of 0.74')
ax.plot(df['Frame'], df['Sharpness 215'], color='blue', label='Sobel predictions \nwith threshold 215')
ax.plot(df_gold['frame_idx'], df_gold['combined'], color='green', label='Gold Standard')

# Set the x-axis limits to 300 and 650
ax.set_xlim(400, 1100)

# Add a legend for the lines
ax.legend(loc='upper left')

# Add a title and axis labels
ax.set_title('Predictions for camera 2\nWindtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3', fontsize=20, fontweight='bold')
ax.set_xlabel('Frame', fontsize=16, fontweight='bold')
ax.set_ylabel('')
ax.set_yticks([0, 1], ['Out of Focus', 'In Focus'])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# Save the figure as an image file
fig.savefig('D:/BAT/Gold/Standard/Combined_images/Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4_camera2_alt.png')

# Display the figure
plt.show()