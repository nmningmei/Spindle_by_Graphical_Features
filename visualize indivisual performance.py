# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:56:29 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score,confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':22})
matplotlib.rcParams['legend.numpoints'] = 1
import seaborn as sns

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
    
signal_features_indivisual_results=pd.read_csv(file_dir+'individual_signal_feature_RF.csv')
graph_features_indivisual_results=pd.read_csv(file_dir+'individual_graph_feature_RF.csv') 

g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results)
g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance\nRandom Forest, 50 estimators')
