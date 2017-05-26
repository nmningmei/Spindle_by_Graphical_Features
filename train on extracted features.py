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
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

if False:
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
        pickle.dump(signal_features_dict,open(file_dir+'signal_features_data.p','wb'))
        pickle.dump(graph_features_dict, open(file_dir+'graph_features_data.p','wb'))
else:
    signal_features_dict=pickle.load(open(file_dir+'signal_features_data.p','rb'))
    graph_features_dict= pickle.load(open(file_dir+'graph_features_data.p','rb'))

##### logitsic regression models ##########
plt.close('all')
keys=['epoch_length 1.5', 'epoch_length 2.0', 
      'epoch_length 2.5', 'epoch_length 3.0', 
      'epoch_length 3.5', 'epoch_length 4.0', 
      'epoch_length 4.5', 'epoch_length 5.0']
"""
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,subtitle='Signal features only, logistic regression')
fig.tight_layout(pad=3.5)
fig.savefig(file_dir+'results\\signal_feature_sum_results.png')

fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,subtitle='Graph features only, logistic regression')
fig.tight_layout(pad=3.5)
fig.savefig(file_dir+'results\\graph_feature_sum_results.png')
"""
####### svm models  rbf kernel ################
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='svm',kernel='rbf',
                                                        subtitle='Signal features only, SVM, kernal: rbf')

