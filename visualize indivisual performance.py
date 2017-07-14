# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:56:29 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
#from sklearn.model_selection import StratifiedKFold,KFold
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score,confusion_matrix
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
#import eegPipelineFunctions
try:
    file_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
#    file_dir = 'D:\\NING - spindle\\training set\\road_trip_more_channels\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
#    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip_more_channels\\'
    os.chdir(file_dir)
def average(x):
    x = x[1:-1].split(', ')
    x = np.array(x,dtype=float)
    return np.nanmean(x)
figsize = 6    
signal_features_indivisual_results_RF=pd.read_csv(file_dir+'individual_signal_feature_RF.csv')
signal_features_indivisual_results_RF['clf']='Random Forest'
graph_features_indivisual_results_RF=pd.read_csv(file_dir+'individual_graph_feature_RF.csv') 
graph_features_indivisual_results_RF['clf']='Random Forest'
combine_features_indevisual_results_RF=pd.read_csv(file_dir+'individual_combine_feature_RF.csv')
combine_features_indevisual_results_RF['clf']='Random Forest'
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on signal features\nRandom Forest, 50 estimators')
#g.savefig(file_dir + 'results\\'+'RF performance signal feature individual.png')
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=graph_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on graph features\nRandom Forest, 50 estimators')
#g.savefig(file_dir + 'results\\'+'RF performance graph feature individual.png')


signal_features_indivisual_results_svm=pd.read_csv(file_dir+'individual_signal_feature_svm.csv')
signal_features_indivisual_results_svm['clf']='Support Vector Machine'
graph_features_indivisual_results_svm=pd.read_csv(file_dir+'individual_graph_feature_svm.csv') 
graph_features_indivisual_results_svm['clf']='Support Vector Machine'
combine_features_indivisual_results_svm=pd.read_csv(file_dir+'individual_combine_feature_svm.csv')
combine_features_indivisual_results_svm['clf']='Support Vector Machine'
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on signal features\nSVM, RBF kernel')
#g.savefig(file_dir + 'results\\'+'svm performance signal feature individual.png')
#g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=graph_features_indivisual_results,size=figsize)
#g.set(xlabel='Epoch legnth',ylabel='Mean AUC scores',title='Classification Performance on graph features\nSVM, RBR kernel')
#g.savefig(file_dir + 'results\\'+ 'svm performance graph feature individual.png')
signal_features_indivisual_results_xgb=pd.read_csv(file_dir+'individual_signal_feature_xgb.csv')
signal_features_indivisual_results_xgb['clf']='X Gredient Boost'
graph_features_indivisual_results_xgb=pd.read_csv(file_dir+'individual_graph_feature_xgb.csv') 
graph_features_indivisual_results_xgb['clf']='X Gredient Boost'
combine_features_indevisual_results_xgb=pd.read_csv(file_dir+'individual_combine_feature_xgb.csv')
combine_features_indevisual_results_xgb['clf']='X Gredient Boost'

signal_features_indivisual_results_knn=pd.read_csv(file_dir+'individual_signal_feature_knn.csv')
signal_features_indivisual_results_knn['clf']='K-nearest Neighbors'
graph_features_indivisual_results_knn=pd.read_csv(file_dir+'individual_graph_feature_knn.csv') 
graph_features_indivisual_results_knn['clf']='K-nearest Neighbors'
combine_features_indevisual_results_knn=pd.read_csv(file_dir+'individual_combine_feature_knn.csv')
combine_features_indevisual_results_knn['clf']='K-nearest Neighbors'

signal_features_indivisual_results_logistic=pd.read_csv(file_dir+'individual_signal_feature_logistic.csv')
signal_features_indivisual_results_logistic['clf']='Logistic regression'
graph_features_indivisual_results_logistic=pd.read_csv(file_dir+'individual_graph_feature_logistic.csv')
graph_features_indivisual_results_logistic['clf']='Logistic regression'
combine_features_indevisual_results_logistic=pd.read_csv(file_dir+'individual_combine_feature_logistic.csv')
combine_features_indevisual_results_logistic['clf']='Logistic regression'



signal_features_indivisual_results = pd.concat([signal_features_indivisual_results_RF,signal_features_indivisual_results_xgb,
                                                signal_features_indivisual_results_logistic,signal_features_indivisual_results_svm,
                                                signal_features_indivisual_results_knn],axis=0)
#                                                signal_features_indivisual_results_logistic],axis=0)
graph_features_indivisual_results = pd.concat([graph_features_indivisual_results_RF,graph_features_indivisual_results_xgb,
                                               graph_features_indivisual_results_logistic,graph_features_indivisual_results_svm,
                                               signal_features_indivisual_results_knn],axis=0)
#                                               graph_features_indivisual_results_logistic],axis=0)

signal_features_indivisual_results['data']='signal'
graph_features_indivisual_results['data']='graph'
#signal_features_indivisual_results = signal_features_indivisual_results.sort_values(['subject','day'])
#graph_features_indivisual_results = graph_features_indivisual_results.sort_values(['subject','day'])
df = pd.concat([signal_features_indivisual_results,graph_features_indivisual_results],axis=0)
signal_features_indivisual_results = signal_features_indivisual_results.drop('data')
signal_features_indivisual_results['auc_score_mean_graph']=graph_features_indivisual_results['auc_score_mean'] # need for the first 2 figures
signal_features_indivisual_results['matthews_corrcoef_mean_graph']=graph_features_indivisual_results['matthews_corrcoef_mean']

#######################################################################################################################################
########################################## AUC factor plot ###########################################################################
orders = ['Random Forest','X Gredient Boost','K-nearest Neighbors',
          'Support Vector Machine','Logistic regression']
g = sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',row='data',
                   col='clf',data=df,col_order=orders)
#for ii, ax in enumerate(g.axes.flatten()):
#    if ii > len(g.axes.flatten()) / 2:
#        ax.set(xticks=[1.0,3.0,5.0],)
#               xticklabels=[1.0,1.6,2.0,2.6,3.0,3.6,4.0,4.6,5.0])
g.fig.savefig(file_dir+'results\\auc factor plot.png',dpi=500)
######################################### MCC factor plot ############################################################################
g = sns.factorplot(x='epoch_length',y='matthews_corrcoef_mean',hue='day',row='data',
                   col='clf',data=df,col_order=orders)
g.fig.savefig(file_dir+'results\\mcc factor plot.png',dpi=500)
#################################################################################################
nnn = 5

grid= sns.FacetGrid(signal_features_indivisual_results,
                    row='epoch_length',col='clf',
                    hue='day',size=2.5,aspect=1.5)
#xx = np.mean(signal_features_indivisual_results['auc_score_mean'])
#yy = np.mean(signal_features_indivisual_results['auc_score_mean_graph'])
#xx_se = np.mean(signal_features_indivisual_results['auc_score_mean'])/ np.sqrt(len(signal_features_indivisual_results))
#yy_se = np.mean(signal_features_indivisual_results['auc_score_mean_graph'])/ np.sqrt(len(signal_features_indivisual_results))
grid.map(plt.hlines,y=0.5,xmin=0,xmax=0.5,color='black',linestyle='--',alpha=0.3)
grid.map(plt.vlines,x=0.5,ymin=0,ymax=0.5,color='black',linestyle='--',alpha=0.3)
#grid.map(plt.errorbar,x=xx,y=yy,xerr=xx_se,yerr=yy_se,color='black',ls='None',alpha=1.0)
grid.map(plt.plot,"auc_score_mean","auc_score_mean_graph",marker='o',ms=7,ls='None',alpha=0.7)
grid.set_axis_labels('AUC of signal features','AUC of graph features')
grid.set(xlim=(0.,1),ylim=(0,1))
T = len(grid.axes.flatten())
for ii,ax in enumerate(grid.axes.flatten()):
    ax.plot([0,1],[0,1],color='navy',linestyle='--')
    #ax.legend(loc='upper left',shadow=False,title='Experiment day')
#    ax.set(xlim=(0,1),ylim=(0,1))
#    if (ii == T) or (ii == T-1) or (ii == T-2):
#        ax.set(xlabel='AUC of signal features',xticks=[0,0.2,0.4,0.6,0.8,1.0],
#               xticklabels=[0,0.2,0.4,0.6,0.8,1.0])
#    elif ii % nnn == 0:
#        ax.set(ylabel='AUC of graph features')
# grid.axes[-1][0].set(ylabel='AUC of graph features',xlabel='AUC of signal features')
grid.savefig(file_dir +'results\\individual performance comparison (AUC).png',dpi=500)



grid= sns.FacetGrid(signal_features_indivisual_results,
                    row='epoch_length',col='clf',
                    hue='day',size=2.5,aspect=1.5)
grid.map(plt.hlines,y=0,xmin=-1,xmax=0,color='black',linestyle='--',alpha=0.3)
grid.map(plt.vlines,x=0,ymin=-1,ymax=0,color='black',linestyle='--',alpha=0.3)
#grid.map(plt.errorbar,x=xx,y=yy,xerr=xx_se,yerr=yy_se,color='black',ls='None',alpha=1.0)
grid.map(plt.plot,"matthews_corrcoef_mean","matthews_corrcoef_mean_graph",marker='o',ms=7,ls='None',alpha=0.7)
grid.set_axis_labels('MCC of signal features','MCC of graph features')
grid.set(xlim=(-1.,1),ylim=(-1,1))
T = len(grid.axes.flatten())
for ii,ax in enumerate(grid.axes.flatten()):
    ax.plot([-1,1],[-1,1],color='navy',linestyle='--')
    #ax.legend(loc='upper left',shadow=False,title='Experiment day')
#    ax.set(xlim=(-1,1),ylim=(-1,1))
#    if (ii == T) or (ii == T-1) or (ii == T-2):
#        ax.set(xlabel='MCC of signal features',xticks=[0,0.2,0.4,0.6,0.8,1.0],
#               xticklabels=[0,0.2,0.4,0.6,0.8,1.0])
#    elif ii % nnn == 0:
#        ax.set(ylabel='MCC of graph features')
#grid.axes[-1][0].set(ylabel='MCC of graph features',xlabel='MCC of signal features')
grid.savefig(file_dir +'results\\individual performance comparison (MCC).png',dpi=500)




a = signal_features_indivisual_results['area_under_precision_recall'].apply(average)
b = graph_features_indivisual_results['area_under_precision_recall'].apply(average)
signal_features_indivisual_results['signal precision-recall score']=a
signal_features_indivisual_results['graph precision-recall score']=b
grid= sns.FacetGrid(signal_features_indivisual_results,
                    row='epoch_length',col='clf',
                    hue='day',size=2.5,aspect=1.5)
grid.map(plt.hlines,y=0.5,xmin=0,xmax=0.5,color='black',linestyle='--',alpha=0.3)
grid.map(plt.vlines,x=0.5,ymin=0,ymax=0.5,color='black',linestyle='--',alpha=0.3)
grid.map(plt.plot,"signal precision-recall score","graph precision-recall score",marker='o',ms=7,ls='None',
         alpha=0.7)
grid.set_axis_labels('precision-recall \nscore of signal features','precision-recall \nscore of graph features')
grid.set(xlim=(0,1),ylim=(0,1))
T = len(grid.axes.flatten())
for ii,ax in enumerate(grid.axes.flatten()):
    ax.plot([0,1],[0,1],color='navy',linestyle='--')
#    ax.set(xlim=(0,1),ylim=(0,1))
#    if (ii == T) or (ii == T-1):# or (ii == T-2):
#        ax.set(xlabel='precision-recall score of signal features')
#    elif ii % nnn == 0:
#        ax.set(ylabel='precision-recall \nscore of graph features')
#grid.axes[-1][0].set(ylabel='precision-recall \nscore of graph features',
#         xlabel='precision-recall score of signal features')
grid.savefig(file_dir +'results\\individual performance comparison precisio-recall.png',dpi=500)

############## TPOT is doing a bad job #####################################################################
#signal_features_indivisual_results_TPOT = pd.read_csv(file_dir+'individual_signal_feature_TPOT.csv')
#graph_features_indivisual_results_TPOT = pd.read_csv(file_dir+'individual_graph_feature_TPOT.csv')
#sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=signal_features_indivisual_results_TPOT)
#sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',data=graph_features_indivisual_results_TPOT)
###############################################################################################################
#
#g=sns.factorplot(x='epoch_length',y='auc_score_mean',hue='day',row='data',col='clf',data=df)
#g.savefig(file_dir+'results\\factor plot of auc.png')
#
#a = df['area_under_precision_recall'].apply(average)
#df['precision-recall score']=a
#g=sns.factorplot(x='epoch_length',y='precision-recall score',hue='day',row='data',col='clf',data=df)
#g.map(plt.hlines,y=0.5,xmin=0,xmax=5,linestyle='--',color='black',alpha=0.7,label='Chance level')
#g.savefig(file_dir+'results\\factor plot of precison-recall score.png')



