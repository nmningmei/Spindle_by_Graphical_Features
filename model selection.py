# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:30:17 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
from tpot import TPOTClassifier
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
if True:
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
            cc_features = cc_features.drop('cc_label',1)
            pli_features = pli_features.drop('pli_label',1)
            plv_features = plv_features.drop('plv_label',1)
            df_combine = pd.concat([cc_features,pli_features,plv_features],axis=1)
            df_combine['label']=label
            df_signal.append(signal_features)
            df_graph.append(df_combine)
        signal_features_dict[directory_1] = pd.concat(df_signal)
        graph_features_dict[directory_1]  = pd.concat(df_graph)
    import pickle
    pickle.dump(signal_features_dict,open(file_dir+'signal features.p','wb'))
    pickle.dump(graph_features_dict,open(file_dir+'graph features.p','wb'))
else:
    import pickle
    signal_features_dict = pickle.load(open(file_dir+'signal features.p','rb'))
    graph_features_dict =  pickle.load(open(file_dir+'graph features.p','rb'))
try:
    file_dir = 'D:\\NING - spindle\\training set\\road_trip\\models\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\models\\'
    os.chdir(file_dir)
for key,dfs in signal_features_dict.items():
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    tpot = TPOTClassifier(generations=10,population_size=25,
                          verbosity=2,random_state=12345,cv=5,scoring='roc_auc')
    tpot.fit(X,Y)
    tpot.score(X,Y)
    tpot.export('%s_%s_tpot_exported_pipeline.py'%('signal_feature',key))
for key,dfs in graph_features_dict.items():
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    tpot = TPOTClassifier(generations=10,population_size=25,
                          verbosity=2,random_state=12345,cv=5,scoring='roc_auc')
    tpot.fit(X,Y)
    tpot.score(X,Y)
    tpot.export('%s_%s_tpot_exported_pipeline.py'%('graph_feature',key))