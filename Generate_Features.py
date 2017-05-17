# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:35:28 2017

@author: ning
"""

import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions

raw_dir = 'D:\\NING - spindle\\training set\\'



raw_files = []
for file in [f for f in os.listdir() if ('txt' in f)]:
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
        

raw_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
    
epoch_lengths  = np.arange(1.5,5.5,0.5) # 1.5 to 5 seconds with 0.5 stepsize
plv_thresholds = np.arange(0.6, 0.85, 0.05) # 0.6 to 0.8 with .05
pli_thresholds = np.arange(0.05,0.30, 0.05) # 0.05 to 0.25 with 0.05
cc_thresholds  = np.arange(0.7, 0.95,0.05) # 0.7 to 0.9 with 0.05

first_level_directory = []
for epoch_length in epoch_lengths:
    directory_1 = raw_dir + 'epoch_length '+str(epoch_length)+'\\'
    if not os.path.exists(directory_1):
        os.makedirs(directory_1)
    first_level_directory.append(directory_1)   


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
        epochs = eegPipelineFunctions.get_data_ready(raw_file,channelList)
        epochFeature = eegPipelineFunctions.featureExtraction(epochs,)
        epochFeature = pd.DataFrame(epochFeature)
        epochFeature.to_csv('sub'+str(sub)+'day'+day+'epoch_features.csv',index=False)
        connectivity = eegPipelineFunctions.connectivity(epochs)
        connectivity = np.array(connectivity)
        plv, pli, cc = connectivity[:,0,:,:],connectivity[:,1,:,:],connectivity[:,2,:,:]
        for t_plv,t_pli,t_cc in zip(plv_thresholds,pli_thresholds,cc_thresholds):
            adj_plv = eegPipelineFunctions.thresholding(t_plv,plv)
            adj_pli = eegPipelineFunctions.thresholding(t_pli,pli)
            adj_cc  = eegPipelineFunctions.thresholding(t_cc, cc )
            graphFeature_plv = eegPipelineFunctions.extractGraphFeatures(adj_plv)
            graphFeature_pli = eegPipelineFunctions.extractGraphFeatures(adj_pli)
            graphFeature_cc  = eegPipelineFunctions.extractGraphFeatures(adj_cc )
            plv_dir = directory_2 + 'plv_' + str(t_plv) + '\\'
            pli_dir = directory_2 + 'pli_' + str(t_pli) + '\\'
            cc_dir  = directory_2 + 'cc_'  + str(t_cc ) + '\\'
            if not os.path.exists(plv_dir):
                os.makedirs(plv_dir)
            if not os.path.exists(pli_dir):
                os.makedirs(pli_dir)
            if not os.path.exists(cc_dir):
                os.makedirs(cc_dir)
            pd.concat([epochFeature,graphFeature_plv],axis=1).to_csv(plv_dir + 'plv_' + str(t_plv) + '.csv',index=False)
            pd.concat([epochFeature,graphFeature_pli],axis=1).to_csv(plv_dir + 'pli_' + str(t_pli) + '.csv',index=False)
            pd.concat([epochFeature,graphFeature_cc ],axis=1).to_csv(cc_dir  + 'cc_'  + str(t_cc ) + '.csv',index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    