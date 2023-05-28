# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:05:11 2023

@author: odens
"""
import os
import pandas as pd

# Set the path of the folder containing the CSV files
path = "D:/BAT/Gold/Standard"

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(path) if f.endswith('combined.csv')]

values = []

for file in csv_files:
    df = pd.read_csv(f'{path}/{file}')
    name = file.partition('_focus_combined.csv')[0]
    frames = len(df['frame_idx'])
    f = (df['focus_Felix'] == 1).sum()
    j = (df['focus_John'] == 1).sum()
    c = (df['combined'] == 2).sum()
    values.append({'Name' : name, 'Felix':f, 'John': j, 'Combined': c, 'Frames':frames})
    

#%% Look at results

df = pd.DataFrame(values, index=None)

df['John_FP'] = df['John'] - df['Combined'] 
df['Felix_FP'] = df['Felix'] - df['Combined']
df['TN'] = df['Frames'] - df['John_FP'] - df['Felix_FP'] - df['Combined']
# Calculate relative observed agreement
df['po'] = (df['Combined'] + df['TN'])/(df['Combined'] + df['TN'] + df['John_FP'] + df['Felix_FP'])
# Calculate expected agreement
df['p1+'] = df['John']/(df['Combined'] + df['TN'] + df['John_FP'] + df['Felix_FP'])
df['p2+'] = df['Felix']/(df['Combined'] + df['TN'] + df['John_FP'] + df['Felix_FP'])
df['p1-'] = (df['Felix_FP'] + df['TN'])/(df['Combined'] + df['TN'] + df['John_FP'] + df['Felix_FP'])
df['p2-'] = (df['John_FP'] + df['TN'])/(df['Combined'] + df['TN'] + df['John_FP'] + df['Felix_FP'])
df['pe'] = (df['p1+'] * df['p2+']) + (df['p1-'] * df['p2-'])
# Calculate Cohen's kappa coefficient

df['Kappa Coefficient'] = (df['po'] - df['pe'])/(1-df['pe'])

# df.to_json(f'{path}/Kappa.json', orient = 'records')
df.to_csv(f'{path}/Kappa.csv', index=False)

df[['Name', 'Kappa Coefficient']].to_csv(f'{path}/just_kappa.csv', index=False)

# Create kappa - coefficient table
#%% Plotting coefficient - names are just too long to fit
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('D:/BAT/Gold/Standard/Kappa.csv')

plt.rcParams.update({'figure.autolayout': True})
fig, ax = plt.subplots()
ax.barh(df['Name'], round(df['Kappa Coefficient'],2))

plt.show()

#%% creating a table using matplotlib

df2 = df[['Name', 'Kappa Coefficient']] # vælg de to kolonner
styler = df2.style # opret en styler objekt
styler.set_properties(**{'text-align': 'center'}) # centrer teksten i cellerne
styler.to_html(f'{path}/table.html') # eksporter som html-fil
