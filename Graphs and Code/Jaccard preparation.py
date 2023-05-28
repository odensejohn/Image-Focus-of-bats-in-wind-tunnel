# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:26:40 2023

@author: odens
"""

import numpy as np
import os
import pandas as pd


#%% Data Preparation
def map_normalised_values(x, threshold):
    if x['Normalised_Camera_1_Face_Sharpness'] and x['Normalised_Camera_2_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0
def map_normalised_camera1_values(x, threshold):
    if x['Normalised_Camera_1_Face_Sharpness']  > threshold:
        return 1
    else:
        return 0

def map_normalised_camera2_values(x, threshold):
    if x['Normalised_Camera_2_Face_Sharpness']  > threshold:
        return 1
    else:
        return 0
    
def map_sharpness_values(x, threshold):
    if x['Camera_1_Face_Sharpness'] and x['Camera_2_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0

def map_sharpness_camera1_values(x, threshold):
    if x['Camera_1_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0

def map_sharpness_camera2_values(x, threshold):
    if x['Camera_2_Face_Sharpness'] > threshold:
        return 1
    else:
        return 0

# Drop unnecessary columns

drop = ['Camera_1_sobel',
        'Camera_1_laplacian',
        'Camera_2_sobel',
       'Camera_2_laplacian',
       'Normalised_Camera_1_laplacian',
       'Normalised_Camera_2_laplacian',
       'Camera_1_Face_Blurriness',
       'Camera_1_x1',
       'Camera_1_y1',
       'Camera_1_x2',
       'Camera_1_y2',
       'Camera_1_BB Confidence',
       'Camera_2_Face_Blurriness',
       'Camera_2_x1',
       'Camera_2_y1',
       'Camera_2_x2',
       'Camera_2_y2',
       'Camera_2_BB Confidence',
       'Deeppose Prediction Camera 1',
       'Deeppose Prediction Camera 2',
       'Camera_1_Width',
       'Camera_1_Height',
       'Camera_1_Area',
       'Camera_1_Sharpness per pixel',
       'Camera_1_Normalised Sharpness per pixel',
       'Camera_2_Width',
       'Camera_2_Height',
       'Camera_2_Area',
       'Camera_2_Sharpness per pixel',
       'Camera_2_Normalised Sharpness per pixel']
    
folder_path = 'D:/BAT/Gold'
gold_path = 'D:/BAT/Gold/Standard'
threshold_values = []
for i in np.arange(0.7, 0.91, 0.01):
    i = round(i, 2)
    threshold_values.append(i)
    

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
        # Adjust json_folder name according to which model's results to use
        json_folder = "D:/BAT/Gold/Yolo2_results"
        output_folder = f'{json_folder}/{NAME}'
        # Adjust dataframe here depending on which model has been used.
        #df = pd.read_json(f'{output_folder}/{NAME}_imadjust_Data.json')
        df = pd.read_json(f'{output_folder}/{NAME}_imadjust_YOLO2_Data.json')
        df.drop(drop, axis=1, inplace=True)
        # Calculate threshold values for combined normalised results
        for i in threshold_values:
            df[f'combined_{i}'] = df.apply(lambda x: map_normalised_values(x, i), axis=1)
        # Calculate threshold values for camera 1 normalised results
        for i in threshold_values:
            df[f'camera1_{i}'] = df.apply(lambda x: map_normalised_camera1_values(x, i), axis=1)
        # Calculate threshold values for camera 2 normalised values
        for i in threshold_values:
            df[f'camera2_{i}'] = df.apply(lambda x: map_normalised_camera2_values(x, i), axis=1)  
        # Calculate threshold values for combined sharpness
        for i in np.arange(150, 225, 5):
              df[f'sharpness_{i}'] = df.apply(lambda x: map_sharpness_values(x, i), axis=1) 
        # threshold values for camera 1 sharpness
        for i in np.arange(150, 225, 5):
              df[f'sharpness1_{i}'] = df.apply(lambda x: map_sharpness_camera1_values(x, i), axis=1)
        # threshold values for camera 2 sharpness
        for i in np.arange(150, 225, 5):
              df[f'sharpness2_{i}'] = df.apply(lambda x: map_sharpness_camera2_values(x, i), axis=1) 
        # Import gold standard values
        df_gold = pd.read_json(f'{gold_path}/{NAME}_Gold.json')
        
        # Merge the values and create new df 
        df_threshold = pd.merge(df_gold, df, 
                                left_on='frame_idx',
                                right_on='Frame',
                                how='inner')
        
         
        df_threshold.to_json(f'{output_folder}/{NAME}_Thresholds.json', orient='records')
        
#%% Calculat Jaccard distances
from scipy.spatial import distance
import json
folder_path = 'D:/BAT/Gold'
jaccard_results = []
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
        # Adjust json_folder name according to which model's results to use
        json_folder = "D:/BAT/Gold/Yolo2_results"
        output_folder = f'{json_folder}/{NAME}'
        
        jaccard_json = []
        df = pd.read_json(f'{output_folder}/{NAME}_Thresholds.json')
        
        keep = []
        drop_columns = ['frame_idx', 'Frame', 'Camera_1_Face_Sharpness',
               'Camera_2_Face_Sharpness', 'Normalised_Camera_1_Face_Sharpness',
               'Normalised_Camera_2_Face_Sharpness' ]
        
        for column in df.columns:
            if column not in drop_columns:
                keep.append(column)
                
        for column in keep:
            jd = distance.jaccard(df['Gold'], df[column])
            jaccard_json.append({'Name':NAME, column : jd})
            
        result = {}
        for d in jaccard_json:
            result.update(d)
        
        jaccard_results.append(result)
        
with open(f'{folder_path}/Yolo2_jaccard_results.json', 'w') as f:
    json.dump(jaccard_results, f)
    
#%% Adding the Average jaccard result to the results - something is wrong in  the state of denmark......

df = pd.read_json(f'{folder_path}/Yolo_jaccard_results.json')
# Subsetting the gold standard information

df_sharp2 = df.loc[[1,6,7,8]]
df_sharp2 = df_sharp2.filter(regex='sharpness_.*')

mean = df_sharp2.mean(numeric_only=True)

df_sharp2.loc['mean'] = mean

df_sharp2.to_json(f'{folder_path}/Yolo_sharpgoldjaccard_results.json')
#%% Plotting Sharpness values for Camera 2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo_sharp2jaccard_results.json')
# create a dictionary to map old column names to new names
column_names = {col: col.split('_')[1] for col in df.columns}

# rename the columns using the dictionary
df = df.rename(columns=column_names)

row = df.iloc[-1]
row= row.round(2)
df2 = pd.read_json('D:/BAT/Gold/Yolo2_sharp2jaccard_results.json')

row2 = df2.iloc[-1]
row2 = row2.round(2)

# Plot the row as a bar plot, excluding the 'Name' column
fig, (ax, ax2)= plt.subplots(nrows=2, figsize=(20,15))
values = row.values
min_indices = np.argwhere(values == np.min(values)).flatten()
colors = ['blue' if i in min_indices else 'grey' for i in range(len(values))]
ax.bar(range(len(df.columns)), values, color=colors)
ax.set_title('Model 1 Mean Jaccard Distance for Variance of the Laplacian scores')
# Add labels inside the bars
for i, v in enumerate(values):
    ax.text(i, v-0.2, str(v), color='white', ha='center', va='center')
ax.set_xticklabels([])
ax.set_ylabel('Jaccard Distance')
# subplot 2
values2 = row2.values
min_indices2 = np.argwhere(values2 == np.min(values2)).flatten()
colors2 = ['blue' if i in min_indices2 else 'grey' for i in range(len(values2))]
ax2.bar(range(len(df.columns)), values2, color=colors2)
ax2.set_title('Model 2 Mean Jaccard Distance for Variance of the Laplacian scores')
# Add labels inside the bars
for i, v in enumerate(values2):
    ax2.text(i, v-0.2, str(v), color='white', ha='center', va='center')
# Set the x-axis tick locations and labels
ax2.set_xticks(range(len(df.columns)))
ax2.set_xticklabels(df.columns)
ax2.set_xlabel('Sharpness Threshold')
ax2.set_ylabel('Jaccard Distance')
# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()
plt.savefig('D:/BAT/Gold/camera2_sharpness_jaccard.png')

#%% Plot Scores for Normalised combined videos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo_goldjaccard_results.json')
# create a dictionary to map old column names to new names
column_names = {col: col.split('_')[1] for col in df.columns}

# rename the columns using the dictionary
df = df.rename(columns=column_names)

row = df.iloc[-1]
row= row.round(2)
df2 = pd.read_json('D:/BAT/Gold/Yolo2_goldjaccard_results.json')

row2 = df2.iloc[-1]
row2 = row2.round(2)

# Plot the row as a bar plot, excluding the 'Name' column
fig, (ax, ax2)= plt.subplots(nrows=2, figsize=(20,15))
values = row.values
min_indices = np.argwhere(values == np.min(values)).flatten()
colors = ['blue' if i in min_indices else 'grey' for i in range(len(values))]
ax.bar(range(len(df.columns)), values, color=colors)
ax.set_title('Model 1 Mean Jaccard Distance for Normalised Sobel scores')
# Add labels inside the bars
for i, v in enumerate(values):
    ax.text(i, v-0.2, str(v), color='white', ha='center', va='center')
ax.set_xticklabels([])
ax.set_ylabel('Jaccard Distance')
# subplot 2
values2 = row2.values
min_indices2 = np.argwhere(values2 == np.min(values2)).flatten()
colors2 = ['blue' if i in min_indices2 else 'grey' for i in range(len(values2))]
ax2.bar(range(len(df.columns)), values2, color=colors2)
ax2.set_title('Model 2 Mean Jaccard Distance for Normalised Sobel scores')
# Add labels inside the bars
for i, v in enumerate(values2):
    ax2.text(i, v-0.2, str(v), color='white', ha='center', va='center')
# Set the x-axis tick locations and labels
ax2.set_xticks(range(len(df.columns)))
ax2.set_xticklabels(df.columns)
ax2.set_xlabel('Normalised Sharpness Threshold')
ax2.set_ylabel('Jaccard Distance')
# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)
fig.suptitle('Gold Standard - Normalised Scores')
plt.tight_layout()
plt.show()
plt.savefig('D:/BAT/Gold/gold_normalised_sobel_jaccard.png')

#%% Plot scores for sharpness for combined videos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo_sharpgoldjaccard_results.json')
# create a dictionary to map old column names to new names
column_names = {col: col.split('_')[1] for col in df.columns}

# rename the columns using the dictionary
df = df.rename(columns=column_names)

row = df.iloc[-1]
row= row.round(2)
df2 = pd.read_json('D:/BAT/Gold/Yolo2_sharpgoldjaccard_results.json')

row2 = df2.iloc[-1]
row2 = row2.round(2)

# Plot the row as a bar plot, excluding the 'Name' column
fig, (ax, ax2)= plt.subplots(nrows=2, figsize=(20,15))
values = row.values
min_indices = np.argwhere(values == np.min(values)).flatten()
colors = ['blue' if i in min_indices else 'grey' for i in range(len(values))]
ax.bar(range(len(df.columns)), values, color=colors)
ax.set_title('Model 1 Mean Jaccard Distance for Sobel scores')
# Add labels inside the bars
for i, v in enumerate(values):
    ax.text(i, v-0.2, str(v), color='white', ha='center', va='center')
ax.set_xticklabels([])
ax.set_ylabel('Jaccard Distance')
# subplot 2
values2 = row2.values
min_indices2 = np.argwhere(values2 == np.min(values2)).flatten()
colors2 = ['blue' if i in min_indices2 else 'grey' for i in range(len(values2))]
ax2.bar(range(len(df.columns)), values2, color=colors2)
ax2.set_title('Model 2 Mean Jaccard Distance for Sobel scores')
# Add labels inside the bars
for i, v in enumerate(values2):
    ax2.text(i, v-0.2, str(v), color='white', ha='center', va='center')
# Set the x-axis tick locations and labels
ax2.set_xticks(range(len(df.columns)))
ax2.set_xticklabels(df.columns)
ax2.set_xlabel('Sobel Threshold')
ax2.set_ylabel('Jaccard Distance')
# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)
fig.suptitle('Gold Standard - Sobel Scores')
plt.tight_layout()
plt.show()
plt.savefig('D:/BAT/Gold/gold_sobel_jaccard.png')
#%% Plot normalised scores for camera 2 videos

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo2_results/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3_camera2_jaccard.json')
df['Camera2'] = df['Camera2'].str.replace('camera2_', '')
df2 = pd.read_json('D:/BAT/Gold/Yolo2_results/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3_sharpness2_jaccard.json')
df2['Sharpness2'] = df2['Sharpness2'].str.replace('sharpness2_', '')

y = np.round(np.linspace(0, 0.9, 10),1)
# Plot the row as a bar plot, excluding the 'Name' column
fig, (ax, ax2)= plt.subplots(nrows=2, figsize=(20,15))
x1 = df['Camera2'].values.astype(float)
values = df['Jaccard Distance']
values = np.around(values.values, decimals = 2)
# create a list of colors
colors = ['blue'] * len(x1) 
ax.bar(df['Camera2'], values, color=colors, label= 'Normalised Sobel Score')
#ax.set_title('Normalised Sobel scores', fontsize = 16, fontweight='bold')
# Add labels inside the bars
for i, v in enumerate(values):
    ax.text(i, v-0.2, str(v), color='white', ha='center', va='center')
ax.set_xticklabels(x1)
ax.set_yticks(y)
ax.set_yticklabels(y, fontsize=16)
ax.set_ylabel('Jaccard Distance', fontsize = 16, fontweight='bold')
ax.legend(loc='upper left', fontsize=14)
# subplot 2
x2 = df2['Sharpness2'].values.astype(float)
colors2 = ['blue'] * len(x2)
values2 = df2['Jaccard Distance']
values2 = np.around(values2.values, decimals=2)
ax2.bar(df2['Sharpness2'], values2, color=colors2, label='Sobel Score')
#ax2.set_title('Sobel scores', fontsize = 16, fontweight='bold')
# Add labels inside the bars
for i, v in enumerate(values2):
    ax2.text(i, v-0.2, str(v), color='white', ha='center', va='center')
# Set the x-axis tick locations and labels
ax2.set_xticks(range(len(df2['Sharpness2'])))
ax2.set_xticklabels(x2)
ax2.set_yticks(y)
ax2.set_yticklabels(y, fontsize=16)
ax2.set_xlabel('Threshold', fontsize = 16, fontweight='bold')
ax2.set_ylabel('Jaccard Distance', fontsize = 16, fontweight='bold')
ax2.legend(loc='upper left', fontsize=14)
# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)
fig.suptitle('Actual Jaccard Distances for Camera 2 from \nWindtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3',
             fontsize = 20, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('D:/BAT/Gold/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3_jaccard.png')

#%% Adding precision, recall, f1 and accuracy measures

from scipy.spatial import distance
import json
import os
import pandas as pd

folder_path = 'D:/BAT/Gold'
results = []

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
        # Adjust json_folder name according to which model's results to use
        json_folder = "D:/BAT/Gold/Yolo_results"
        output_folder = f'{json_folder}/{NAME}'
        
        jaccard_json = []
        df = pd.read_json(f'{output_folder}/{NAME}_Thresholds.json')
        # Convert columns to numeric data type
        df = df.apply(pd.to_numeric, errors='coerce')
        gold = df['Gold']
        
        drop_columns = ['frame_idx', 'Frame', 'Camera_1_Face_Sharpness',
                        'Camera_2_Face_Sharpness', 'Normalised_Camera_1_Face_Sharpness',
                        'Normalised_Camera_2_Face_Sharpness', 'Gold']
        
        for column in df.columns:
            if column not in drop_columns:
                predicted = df[column]
                true_positives = ((gold == 1) & (predicted == 1)).sum()
                false_positives = ((gold == 0) & (predicted == 1)).sum()
                false_negatives = ((gold == 1) & (predicted == 0)).sum()
                true_negatives = ((gold == 0) & (predicted == 0)).sum()
                
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                accuracy = (true_positives + true_negatives) / len(gold)
                f1 = 2 * (precision * recall)/(precision + recall)
                jaccard_distance = distance.jaccard(gold, predicted)
                
                jaccard_json.append({
                    'Name': NAME,
                    'Column': column,
                    'Jaccard Distance': jaccard_distance,
                    'Precision': precision,
                    'Recall': recall,
                    'Accuracy': accuracy,
                    'F1': f1
                })
            
        results.extend(jaccard_json)
        
with open(f'{folder_path}/Yolo_4measures_results.json', 'w') as f:
    json.dump(results, f, indent=4)

#%% Import pandas and re modules
import pandas as pd 
import re
import os
df = pd.read_json('D:/BAT/Gold/Yolo_4measures_results.json')
# Define the regex patterns to split by
patterns = ['combined_', 'camera1_', 'camera2', 'sharpness_', 'sharpness1_', 'sharpness2_']

output_path = 'D:\BAT\Gold\Results'
# Loop over the regex patterns 
for pattern in patterns: 
    if not os.path.isdir(f'{output_path}/{pattern}Yolo'):
        os.makedirs(f'{output_path}/{pattern}Yolo')                        
    # Subset the dataframe by the pattern 
    df_pattern = df.loc[df['Column'].str.contains(pattern, flags=re.IGNORECASE)]
    # Round the values in the other columns to 2 decimal places 
    df_pattern = df_pattern.round(2) 
    # Export the dataframe to a csv file with a unique name 
    df_pattern.to_csv(f'{output_path}/{pattern}Yolo/{pattern}.csv', index=False)
    
#%% Calculating mean values
# Drop the name column
import pandas as pd
import json

df = pd.read_json('D:/BAT/Gold/Yolo2_4measures_results.json')
# Get list of names for videos
unique_names = list(df['Name'].unique())

gold_names = ['Windtunnel_Prey_Capture_Approach_1628084756_20210804_154556_S001_Mnat_2021_36_3',
              'Windtunnel_Prey_Capture_Zoom_1631008068_20210907_114748_S001_Mnat_2021_36_8',
              'Windtunnel_Prey_Capture_Zoom_1632316045_20210922_150725_S003_Mnat_2021_36_5',
              'Windtunnel_Prey_Capture_Zoom_1632388467_20210923_111427_S001_Mnat_2021_36_3']

cam2_names = ['Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3',
              'Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4']
# Individual videos for results comparison
cam2_jac1 = ['Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3']
cam2_jac2 = ['Windtunnel_Prey_Capture_Buzz_1629195696_20210817_122136_S002_Mnat_2021_36_4']

# Drop rows that aren't needed
#df = df[df["Name"].isin(gold_names)]
#df = df[df["Name"].isin(cam2_names)]
df = df[df["Name"].isin(cam2_jac1)]
df = df.drop(columns='Name')
# Group by unique column names
groups = df.groupby('Column')

# Calculate mean values
means = groups.mean()
# Returns column names
means = means.reset_index()
# Round to 2 decimal places
means = means.round(2)

records = means.to_dict(orient='records')

#with open('D:/BAT/Gold/Results/Yolo2/means.json', 'w') as f:
#    json.dump(records, f)
    
# Create individual files:

colvalues = ['camera2_','sharpness2_']
# colvalues = ['combined_','sharpness_']
''' - for combined results
for value in colvalues:
  # filter out the rows that contain the value in the Column column
  filtered_df = means[means["Column"].str.contains(value)]
  filtered_df = filtered_df.reset_index()
  filtered_df = filtered_df.drop(columns = 'index')
  # Rename Column
  new_value = value.capitalize().replace('_', '')
  filtered_df = filtered_df.rename(columns={'Column': new_value})
  out = filtered_df.to_dict(orient='records')
  # save the filtered dataframe as a json file with the value as part of the file name
  with open(f'D:/BAT/Gold/Results/Yolo2/{value}mean.json', 'w') as f:
      json.dump(out, f)
      '''
for value in colvalues:
    filtered_df = df[df['Column'].str.contains(value)]
    filtered_df = filtered_df.reset_index()
    filtered_df = filtered_df.drop(columns = 'index')
    new_value = value.capitalize().replace('_', '')
    filtered_df = filtered_df.rename(columns={'Column': new_value})
    out = filtered_df.to_dict(orient='records')
    with open(f'D:/BAT/Gold/Yolo2_results/Windtunnel_Prey_Capture_Approach_1628587895_20210810_113135_S001_Mnat_2021_36_3_{value}jaccard.json', 'w') as f:
        json.dump(out, f)
        
#%% Group bar chart normalised values

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo_goldjaccard_results.json')
# create a dictionary to map old column names to new names
column_names = {col: col.split('_')[1] for col in df.columns}

# rename the columns using the dictionary
df = df.rename(columns=column_names)

row = df.iloc[-1]
row= row.round(2)
df2 = pd.read_json('D:/BAT/Gold/Yolo2_goldjaccard_results.json')

row2 = df2.iloc[-1]
row2 = row2.round(2)

fig, ax = plt.subplots(figsize=(20,15))

x = np.arange(len(row))
y = np.round(np.linspace(0, 0.9, 10),1)
# Set width of bars
width = 0.4

# Create bars for each row
ax.bar(x - width/2, row.values, width, label='Model 1', color='blue')
ax.bar(x + width/2, row2.values, width, label='Model 2', color='orange')

# Set the title and axis labels
ax.set_title('Mean Jaccard Distance for Normalised Sobel scores', fontsize = 20, fontweight='bold')
ax.set_xlabel('Normalised Sharpness Threshold', fontsize = 16, fontweight='bold')
ax.set_ylabel('Jaccard Distance', fontsize = 16, fontweight='bold')

# Set x ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(row.index, fontsize = 16)
ax.set_yticks(y)
ax.set_yticklabels(y, fontsize=16)
# Add a legend
ax.legend(title_fontsize=20, fontsize=16)

# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()

plt.savefig('D:/BAT/Gold/grouped_normalised_jaccard.png')

#%% Grouped Plot sobel values
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting results 
df = pd.read_json('D:/BAT/Gold/Yolo_sharpgoldjaccard_results.json')
# create a dictionary to map old column names to new names
column_names = {col: col.split('_')[1] for col in df.columns}

# rename the columns using the dictionary
df = df.rename(columns=column_names)

row = df.iloc[-1]
row= row.round(2)
df2 = pd.read_json('D:/BAT/Gold/Yolo2_sharpgoldjaccard_results.json')

row2 = df2.iloc[-1]
row2 = row2.round(2)

fig, ax = plt.subplots(figsize=(20,15))

x = np.arange(len(row))
y = np.round(np.linspace(0, 0.9, 10),1)
# Set width of bars
width = 0.4

# Create bars for each row
ax.bar(x - width/2, row.values, width, label='Model 1', color='blue')
ax.bar(x + width/2, row2.values, width, label='Model 2', color='orange')

# Set the title and axis labels
ax.set_title('Mean Jaccard Distance for Sobel scores', fontsize = 20, fontweight='bold')
ax.set_xlabel('Sharpness Threshold', fontsize = 16, fontweight='bold')
ax.set_ylabel('Jaccard Distance', fontsize = 16, fontweight='bold')

# Set x ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(row.index, fontsize = 16)
ax.set_yticks(y)
ax.set_yticklabels(y, fontsize=16)
# Add a legend
ax.legend(title_fontsize=20, fontsize=16, loc='upper left')

# Increase the bottom margin to make room for the labels
fig.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()

plt.savefig('D:/BAT/Gold/grouped_jaccard.png')

#%% Calculating F1 scores 

import pandas as pd

df = pd.read_json('D:/BAT/Gold/Results/Yolo2/complete_mean.json')

df['F1'] = round(2 * (df['Precision']*df['Recall'])/(df['Precision']+df['Recall']),3)

df.to_json('D:/BAT/Gold/Results/Yolo2/complete_sharpness_mean.json', orient = 'records')

