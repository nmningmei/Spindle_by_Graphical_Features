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
annotation_file = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\annotations\\suj11_nap_day2_edited_annotations.txt'
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions

epochs,label,temp = eegPipelineFunctions.get_data_ready(filename,channelList,annotation_file,)

epochFeature = eegPipelineFunctions.featureExtraction(epochs,)
connectivity = eegPipelineFunctions.connectivity(epochs)
t = 0.8
connectivity = np.array(connectivity)
cc = connectivity[:,-1,:,:]

adj = eegPipelineFunctions.thresholding(t,cc)
graphFeature = eegPipelineFunctions.extractGraphFeatures(adj)
before_feature = eegPipelineFunctions.extractGraphFeatures(cc)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,matthews_corrcoef,make_scorer
lcv = np.logspace(-3,3,7)
cv = KFold(n_splits=10,shuffle=True,random_state=123)
clf = Pipeline([('scaler',StandardScaler()),
               ('estimator',LogisticRegressionCV(Cs=lcv,cv=cv,max_iter=1e4))])
#results = []
#for train,test in cv.split(graphFeature.values):
#    x_train = graphFeature.values[train]
#    y_train = label[train]
#    clf.fit(x_train,y_train)
#    x_test = graphFeature.values[test]
#    y_test = label[test]
#    
#    results.append(f1_score(y_test,clf.predict(x_test)))
features = pd.concat([graphFeature,before_feature],axis=1)
results= cross_val_score(clf,features.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
results = cross_val_score(clf,graphFeature.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
results = cross_val_score(clf,before_feature.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
