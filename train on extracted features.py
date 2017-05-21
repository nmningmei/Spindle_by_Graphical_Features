# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:13:26 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
import eegPipelineFunctions
raw_dir = 'D:\\NING - spindle\\training set\\'
# get EEG files that have corresponding annotations
raw_files = []
for file in [f for f in os.listdir(raw_dir) if ('txt' in f)]:
    sub = int(file.split('_')[0][3:])
    if sub < 11:
        day = file.split('_')[1][1]
        day_for_load = file.split('_')[1][:2]
    else:
        day = file.split('_')[2][-1]
        day_for_load = file.split('_')[2]
    raw_file = [f for f in os.listdir(raw_dir) if (file.split('_')[0] in f) and (day_for_load in f) and ('fif' in f)]
    if len(raw_file) != 0:
        raw_files.append([raw_dir + raw_file[0],raw_dir + file])
# directory for storing all the feature files
raw_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
# initialize the range of the parameters we want to compute based on
epoch_lengths  = np.arange(1.5,5.5,0.5) # 1.5 to 5 seconds with 0.5 stepsize
plv_thresholds = np.arange(0.6, 0.85, 0.05) # 0.6 to 0.8 with .05
pli_thresholds = np.arange(0.05,0.30, 0.05) # 0.05 to 0.25 with 0.05
cc_thresholds  = np.arange(0.7, 0.95,0.05) # 0.7 to 0.9 with 0.05
# make sub-directories based on epoch length
first_level_directory = []
for epoch_length in epoch_lengths:
    directory_1 = raw_dir + 'epoch_length '+str(epoch_length)+'\\'
    if not os.path.exists(directory_1):
        os.makedirs(directory_1)
    first_level_directory.append(directory_1)   

# signal features only
signal_feature = {str(epoch_length):[] for epoch_length in epoch_lengths}
for d1 in first_level_directory:
    os.chdir(d1)
    #print(os.getcwd())
    for files in raw_files:
        raw_file, annotation_file = files
        temp_anno = annotation_file.split('\\')[-1]
        sub = int(temp_anno.split('_')[0][3:])
        if sub < 11:
            day = temp_anno.split('_')[1][1]
            day_for_load = temp_anno.split('_')[1][:2]
        else:
            day = temp_anno.split('_')[2][-1]
            day_for_load = temp_anno.split('_')[2]
        directory_2 = d1 + 'sub' + str(sub) + 'day' + day + '\\'
        if not os.path.exists(directory_2):
            #print(directory_2)
            os.makedirs(directory_2)
        os.chdir(directory_2)
        print(directory_2)
        
