# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:55:32 2017

@author: ning
"""
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score
import pickle
import matplotlib.pyplot as plt
try:
    function_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
except:
    function_dir = 'C:\\Users\\ning\\OneDrive\\python works\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
import eegPipelineFunctions
try:
    file_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
    os.chdir(file_dir)
siganl_only_result = pickle.load(open(file_dir+'signal_feature_only.p','rb'))
graph_only_result = pickle.load(open(file_dir+'graph_feature_only.p','rb'))


fig, axes = plt.subplots(4,2,figsize=(8,16))
for ax,(key,values) in zip(axes.flatten(),graph_only_result.items()):
    epoch_length = key
    auc_score,fpr,tpr,precision,recall,average_scores =values[2]
    ax.plot(recall,precision)
    
    
    


