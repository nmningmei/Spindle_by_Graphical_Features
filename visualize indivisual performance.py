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
sns.set_style('white')

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
figsize = 6    
signal_features_indivisual_results_RF=pd.read_csv(file_dir+'individual_signal_feature_RF.csv')
signal_features_indivisual_results_RF['clf']='Random Forest'
graph_features_indivisual_results_RF=pd.read_csv(file_dir+'individual_graph_feature_RF.csv') 
graph_features_indivisual_results_RF['clf']='Random Forest'
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on signal features\nRandom Forest, 50 estimators')
#g.savefig(file_dir + 'results\\'+'RF performance signal feature individual.png')
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=graph_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on graph features\nRandom Forest, 50 estimators')
#g.savefig(file_dir + 'results\\'+'RF performance graph feature individual.png')


signal_features_indivisual_results_svm=pd.read_csv(file_dir+'individual_signal_feature_svm.csv')
signal_features_indivisual_results_svm['clf']='Support Vector Machine'
graph_features_indivisual_results_svm=pd.read_csv(file_dir+'individual_graph_feature_svm.csv') 
graph_features_indivisual_results_svm['clf']='Support Vector machine'
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on signal features\nSVM, RBF kernel')
#g.savefig(file_dir + 'results\\'+'svm performance signal feature individual.png')
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=graph_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on graph features\nSVM, RBR kernel')
#g.savefig(file_dir + 'results\\'+ 'svm performance graph feature individual.png')
signal_features_indivisual_results = pd.concat([signal_features_indivisual_results_RF,signal_features_indivisual_results_svm],axis=0)
graph_features_indivisual_results = pd.concat([graph_features_indivisual_results_RF,graph_features_indivisual_results_svm],axis=0)
signal_features_indivisual_results['auc_score_mean_graph']=graph_features_indivisual_results['auc_score_mean']
#g = sns.factorplot(x='auc_score_mean',y='auc_score_mean_graph',hue='day',row='clf',
#                   col='epoch_length',data=signal_features_indivisual_results)

grid= sns.FacetGrid(signal_features_indivisual_results,row='epoch_length',col='clf',hue='day',size=4)
grid.map(plt.plot, x=[0,1],y=[0,1],color='black',linestyle='--')
grid.map(plt.plot,"auc_score_mean","auc_score_mean_graph",marker='o',ms=7,linestyle='None')
for ii,ax in enumerate(grid.axes.flatten()):
    ax.plot([0,1],[0,1],color='navy',linestyle='--')
    ax.legend(loc='upper left',shadow=False,title='Experiment day')
    if (ii == 14) or (ii == 15):
        ax.set(xlabel='AUC of signal features')
    elif ii % 2 == 0:
        ax.set(ylabel='AUC of graph features')
grid.savefig(file_dir +'results\\individual performance comparison.png')















