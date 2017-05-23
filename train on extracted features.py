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
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score,confusion_matrix
import pickle
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

signal_features_dict = {}
graph_features_dict = {}
for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1[-3]
    os.chdir(sub_dir)
    df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
        #df_cc.append(cc_features)
        #df_pli.append(pli_features)
        #df_plv.append(plv_features)
        label = cc_features['label']
        cc_features = eegPipelineFunctions.get_real_part(cc_features)
        pli_features = eegPipelineFunctions.get_real_part(pli_features)
        plv_features = eegPipelineFunctions.get_real_part(plv_features)
        cc_features.columns = ['cc_'+name for name in cc_features]
        pli_features.columns = ['pli_'+name for name in pli_features]
        plv_features.columns = ['plv_'+name for name in plv_features]
        df_combine = pd.concat([cc_features.iloc[:,:-1],pli_features.iloc[:,:-1],plv_features.iloc[:,:-1]],axis=1)
        df_combine['label']=label
        df_signal.append(signal_features)
        df_graph.append(df_combine)
    signal_features_dict[directory_1] = pd.concat(df_signal)
    graph_features_dict[directory_1]  = pd.concat(df_graph)
results = {}
for key,dfs in signal_features_dict.items():
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=np.random.randint(10000,20000))
    results[key] = []
    for train, test in cv.split(X,Y):
        clf = Pipeline([('scaler',StandardScaler()),
                        ('estimator',LogisticRegressionCV(Cs=np.logspace(-3,3,7),
                                                          max_iter=int(1e5),
                                                          tol=1e-4,
                                                          cv=KFold(n_splits=10,shuffle=True,random_state=12345),
                                                          class_weight={1:1-np.count_nonzero(Y)/len(Y),0:(np.count_nonzero(Y)/len(Y))},
                                                          scoring='roc_auc'))])
        clf.fit(X[train],Y[train])
        fpr,tpr,_ = roc_curve(Y[test],clf.predict_proba(X[test])[:,-1])
        auc_score = auc(fpr,tpr)
        precision,recall,_ = precision_recall_curve(Y[test],clf.predict_proba(X[test])[:,-1])
        average_scores = average_precision_score(Y[test],clf.predict_proba(X[test])[:,-1])
        results[key].append([auc_score,fpr,tpr,precision,recall,average_scores])
        print(key,auc_score,precision,recall,average_scores,'signal\n',confusion_matrix(Y[test],clf.predict(X[test])))
pickle.dump(results,open(file_dir+'signal_feature_only.p','wb'))

results = {}
for key,dfs in graph_features_dict.items():
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=np.random.randint(10000,20000))
    results[key] = []
    for train, test in cv.split(X,Y):
        clf = Pipeline([('scaler',StandardScaler()),
                        ('estimator',LogisticRegression(C=1e2,
                                                          max_iter=int(1e5),
                                                          tol=1e-4))])
        clf.fit(X[train],Y[train])
        fpr,tpr,_ = roc_curve(Y[test],clf.predict_proba(X[test])[:,-1])
        auc_score = auc(fpr,tpr)
        precision,recall,_ = precision_recall_curve(Y[test],clf.decision_function(X[test]))
        average_scores = average_precision_score(Y[test],clf.decision_function(X[test]))
        results[key].append([auc_score,fpr,tpr,precision,recall,average_scores])
        print(key,auc_score,precision,recall,average_scores,'graph\n',confusion_matrix(Y[test],clf.predict(X[test])))
pickle.dump(results,open(file_dir+'graph_feature_only.p','wb'))         
    
#siganl_only_result = pickle.load(open(file_dir+'signal_feature_only.p','rb'))
#graph_only_result = pickle.load(open(file_dir+'graph_feature_only.p','rb'))
