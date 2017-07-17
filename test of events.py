# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:17:08 2017

@author: ning
"""

#import mne
#import numpy as np
#import pandas as pd
import os
#import networkx as nx
#from collections import Counter
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
channelList = ['F3','F4','C3','C4','O1','O2']
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
        
raw_file, annotation_file = raw_files[0]
l_freq = 11;h_freq = 16
epoch_length = 2; overlapping = 0.2
#raw = mne.io.read_raw_fif(raw_file,preload=True)
#if channelList is not None:
#    raw.pick_channels(channelList)
#else:
#    raw.drop_channels(['LOc','ROc'])
#raw.filter(l_freq,h_freq,filter_length='10s', l_trans_bandwidth=0.1, h_trans_bandwidth=0.5,n_jobs=4,)
#a=epoch_length - overlapping * 2
#events = mne.make_fixed_length_events(raw,id=1,duration=a)
#epochs = mne.Epochs(raw,events,tmin=0,tmax=epoch_length,baseline=None,preload=True,proj=False)
#epochs.resample(100)
#annotation = pd.read_csv(annotation_file)
#spindles = annotation[annotation['Annotation'].apply(eegPipelineFunctions.spindle_check)]
epochs,label,_ = eegPipelineFunctions.get_data_ready(raw_file,channelList,
                                                             annotation_file,
                                                             epoch_length=epoch_length)
