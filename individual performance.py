# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:52:15 2017

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
import pickle
from collections import Counter

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
#siganl_only_result = pickle.load(open(file_dir+'signal_feature_only.p','rb'))
#graph_only_result = pickle.load(open(file_dir+'graph_feature_only.p','rb'))
signal_features_dict = {}
graph_features_dict = {}
for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1[-3]
    os.chdir(sub_dir)
    signal_features_dict[directory_1],graph_features_dict[directory_1]={},{}
    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        sub = sub_fold[:4]
        day = sub_fold[4:]
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
        try:
            signal_features_dict[directory_1][sub_fold] = eegPipelineFunctions.cross_validation_pipeline(signal_features)
            graph_features_dict[directory_1][sub_fold] = eegPipelineFunctions.cross_validation_pipeline(df_combine)
            print(sub_fold,Counter(label),'enough samples??')
            print('signal:%.2f +/-%.2f'%(np.mean([v for (v,_,_,_) in signal_features_dict[directory_1][sub_fold]]),np.std([v for (v,_,_,_) in signal_features_dict[directory_1][sub_fold]])))
            print('graph:%.2f +/-%.2f'%(np.mean([v for (v,_,_,_) in graph_features_dict[directory_1][sub_fold]]),np.std([v for (v,_,_,_) in graph_features_dict[directory_1][sub_fold]])))
            #print(confusion_matrix())
        except:
            try:
                cv = KFold(n_splits=5,shuffle=True,random_state=0)
                signal_features_dict[directory_1][sub_fold] = eegPipelineFunctions.cross_validation_pipeline(signal_features,cv=cv)
                graph_features_dict[directory_1][sub_fold] = eegPipelineFunctions.cross_validation_pipeline(df_combine,cv=cv)
                print(sub_fold,Counter(label),'too few samples??')
            except:
                print(sub_fold,Counter(label),'not enough samples')
pickle.dump(signal_features_dict,open(file_dir+'individual_signal_feature_only.p','wb'))
pickle.dump(graph_features_dict,open(file_dir+'individual_graph_feature_only.p','wb'))