# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:16:30 2023

@author: odens
"""

import pandas as pd
import os

folder_path = 'D:/BAT/Gold'
csv_path = 'D:/BAT/Gold/Standard'

def gold_standard(df):
    if df['combined_x'] == 2 and df['combined_y'] == 2:
        return 1 
    else:
        return 0

for file_name in os.listdir(folder_path):
    if file_name.endswith('.h5'):
    # Import file   
        f5 = f'{folder_path}/{file_name}'
        # Extract the filename from the path
        filename = os.path.basename(f5)
        
        # Split the filename at the last occurrence of "/"
        head, _, tail = filename.rpartition('/')
        
        # Remove the extension ".h5" from the tail
        NAME = tail.rpartition('.')[0]
        
        # Create dataframes
        
        df1 = pd.read_csv(f'{csv_path}/{NAME}_camera_1_focus_combined.csv')
        df2 = pd.read_csv(f'{csv_path}/{NAME}_camera_2_focus_combined.csv')
        
        df_merged = pd.merge(df1, df2, on='frame_idx')
        
        df_merged['Gold'] = df_merged.apply(lambda x: gold_standard(x), axis=1)
        
        df_gold = df_merged[['frame_idx', 'Gold']]
        
        df_gold.to_json(f'{csv_path}/{NAME}_Gold.json')
        
        
