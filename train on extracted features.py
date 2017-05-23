# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:13:26 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score
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
    
#epoch_lengths  = np.arange(1.5,5.5,0.5) # 1.5 to 5 seconds with 0.5 stepsize
#signal_features_dict = {str(epoch_length):[] for epoch_length in epoch_lengths}
signal_features_dict = {}

for directory_1 in os.listdir(file_dir):
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1[-3]
    os.chdir(sub_dir)
    df_cc, df_pli, df_plv, df_signal = [],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
        df_cc.append(cc_features)
        df_pli.append(pli_features)
        df_plv.append(plv_features)
        df_signal.append(signal_features)
    signal_features_dict[directory_1] = pd.concat(df_signal)
results = {}
for key,dfs in signal_features_dict.items():
    data = dfs.values   
    X, Y = data[:,:-2], data[:,-1]
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=np.random.randint(10000,20000))
    results[key] = []
    for train, test in cv.split(X,Y):
        clf = Pipeline([('scaler',StandardScaler()),
                        ('estimator',LogisticRegressionCV(Cs=np.logspace(-3,3,7),
                                                          max_iter=int(1e5),
                                                          tol=1e-4,
                                                          cv=KFold(n_splits=10,shuffle=True,random_state=2017),
                                                          class_weight={1:np.count_nonzero(Y)/len(Y),0:1-(np.count_nonzero(Y)/len(Y))},
                                                          scoring='roc_auc'))])
        clf.fit(X[train],Y[train])
        fpr,tpr,_ = roc_curve(Y[test],clf.predict_proba(X[test])[:,-1],pos_label=1)
        auc_score = auc(fpr,tpr)
        precision,recall,_ = precision_recall_curve(Y[test],clf.predict_proba(X[test])[:,-1],pos_label=1)
        average_scores = average_precision_score(Y[test],clf.predict_proba(X[test])[:,-1],average='samples')
        results[key].append([auc_score,precision,recall,average_scores])
         
    

        
