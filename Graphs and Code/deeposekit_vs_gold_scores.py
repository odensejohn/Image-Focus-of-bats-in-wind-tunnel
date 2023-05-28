# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:21:12 2023

@author: odens
"""

import pandas as pd
import matplotlib.pyplot as plt

dpk_results = []
['Windtunnel_Prey_Capture_Approach_1628084756_20210804_154556_S001_Mnat_2021_36_3',
'Windtunnel_Prey_Capture_Zoom_1631008068_20210907_114748_S001_Mnat_2021_36_8',
'Windtunnel_Prey_Capture_Zoom_1632316045_20210922_150725_S003_Mnat_2021_36_5',
'Windtunnel_Prey_Capture_Zoom_1632388467_20210923_111427_S001_Mnat_2021_36_3']
#%%
NAME = 'Windtunnel_Prey_Capture_Zoom_1632388467_20210923_111427_S001_Mnat_2021_36_3'
df = pd.read_json('D:/BAT/Gold/Standard/Windtunnel_Prey_Capture_Zoom_1632388467_20210923_111427_S001_Mnat_2021_36_3_Gold.json')

felix_df = pd.read_csv('D:/BAT/Gold/Felix Predictions/Windtunnel_Prey_Capture_Zoom_1632388467_20210923_111427_S001_Mnat_2021_36_3_dpk_frame_selection.csv')
mask = felix_df.iloc[::15, :]
mask = mask.reset_index(drop=True)

df['cluster_selection'] = mask['cluster_selection']


fig, ax = plt.subplots()
ax.plot(df['frame_idx'], df['cluster_selection'])
ax.plot(df['frame_idx'], df['Gold'])
fig.show()
#%%  Calculating f1, recall, accuracy for deepposekit
from scipy.spatial import distance

gold = df['Gold']
predicted = df['cluster_selection']
true_positives = ((gold == 1) & (predicted == 1)).sum()
false_positives = ((gold == 0) & (predicted == 1)).sum()
false_negatives = ((gold == 1) & (predicted == 0)).sum()
true_negatives = ((gold == 0) & (predicted == 0)).sum()

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
accuracy = (true_positives + true_negatives) / len(gold)
f1 = 2 * (precision * recall)/(precision + recall)
jaccard_distance = distance.jaccard(gold, predicted)

dpk_results.append({
    'Name': NAME,
    'Jaccard Distance': jaccard_distance,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'F1': f1
})
#%% Exporting results to json
import json
with open('D:/BAT/Gold/deeposekit_predictions_vs_gold.json', 'w') as f:
    json.dump(dpk_results, f)