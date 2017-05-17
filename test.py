# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:59:09 2017

@author: ning
"""

import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
filename = 'D:\\NING - spindle\\training set\\suj11_l2nap_day2.fif'
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions

epochs = eegPipelineFunctions.get_data_ready(filename,channelList)
epochFeature = eegPipelineFunctions.featureExtraction(epochs,)
connectivity = eegPipelineFunctions.connectivity(epochs)
t = 0.8
connectivity = np.array(connectivity)
cc = connectivity[:,-1,:,:]
adj = eegPipelineFunctions.thresholding(t,cc)
graphFeature = eegPipelineFunctions.extractGraphFeatures(adj)
