# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:13:26 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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
#    file_dir = 'D:\\NING - spindle\\training set\\road_trip_29_channels\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
#    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip_29_channels\\'
    os.chdir(file_dir)

if False:
    signal_features_dict = {}
    graph_features_dict = {}
    my_features_dict = {}
    for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
        sub_dir = file_dir + directory_1 + '\\'
        epoch_length = directory_1[-3]
        os.chdir(sub_dir)
        df_cc, df_pli, df_plv, df_signal,df_graph,df_my = [],[],[],[],[],[]
        for sub_fold in os.listdir(sub_dir):
            sub_fold_dir = sub_dir + sub_fold + '\\'
            os.chdir(sub_fold_dir)
            dfs = [f for f in os.listdir(sub_fold_dir) if ('csv' in f)]
            signal_features = pd.read_csv([f for f in dfs if ('epoch' in f)][0])
            try:
                df_my_feature = pd.read_csv([f for f in dfs if ('my' in f)][0])
            except:
                pass
            cc_features = pd.read_csv([f for f in dfs if ('cc' in f)][0])
            pli_features = pd.read_csv([f for f in dfs if ('pli' in f)][0])
            plv_features = pd.read_csv([f for f in dfs if ('plv' in f)][0])
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
            try:
                df_my.append(df_my_feature)
            except:
                pass
        signal_features_dict[directory_1] = pd.concat(df_signal)
        graph_features_dict[directory_1]  = pd.concat(df_graph)
        pickle.dump(signal_features_dict,open(file_dir+'signal_features_data.p','wb'))
        pickle.dump(graph_features_dict, open(file_dir+'graph_features_data.p','wb'))
        try:
            my_features_dict[directory_1] = pd.concat(df_my)
            pickle.dump(my_features_dict,open(file_dir+'my_features_data.p','wb'))
        except:
            pass
        
else:
    signal_features_dict=pickle.load(open(file_dir+'signal_features_data.p','rb'))
    graph_features_dict= pickle.load(open(file_dir+'graph_features_data.p','rb'))
    my_features_dict = pickle.load(open(file_dir+'my_features_data.p','rb'))
keys=['epoch_length ' + str(a) for a in np.arange(1.,5.,0.2)]
##### logitsic regression models ##########
plt.close('all')

fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='logistic',subtitle='Signal features only, logistic regression')
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_regression_results.png',dpi=500)

fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='logistic',subtitle='Graph features only, logistic regression')
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_regression_results.png',dpi=500)

####### svm models  rbf kernel ################
plt.close('all')
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='svm',kernel='rbf',
                                                        subtitle='Signal features only, SVM, kernal: rbf',n_estimators=10,bag=True)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_svm_rbf_results.png',dpi=500)

fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='svm',kernel='poly',
                                                        subtitle='Signal features only, SVM, kernal: poly',n_estimators=10,bag=True)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_svm_poly_results.png',dpi=500)

####### svm models graph feature ##########
plt.close('all')
fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='svm',kernel='rbf',
                                                        subtitle='Signal features only, SVM, kernal: rbf',n_estimators=10,bag=True)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_svm_rbf_results.png',dpi=500)

fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='svm',kernel='poly',
                                                        subtitle='graph features only, SVM, kernal: poly',n_estimators=10,bag=True)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_svm_poly_results.png',dpi=500)

#### random forest #######
plt.close('all')
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='RF',
                                                        subtitle='Signal features only, RF, n_estimator:50')
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_RF_results.png',dpi=500)

fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='RF',
                                                        subtitle='graph features only, RF, n_estimator:50')#,weights=8)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_RF_results.png',dpi=500)

#### xgb #######
plt.close('all')
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='xgb',
                                                        subtitle='Signal features only, xgb, n_estimator:100')
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_xgb_results.png',dpi=500)

fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='xgb',
                                                        subtitle='graph features only, xgb, n_estimator:100')#,weights=8)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_xgb_results.png',dpi=500)
#### knn #######
plt.close('all')
fig=eegPipelineFunctions.visualize_auc_precision_recall(signal_features_dict,keys,clf_='knn',
                                                        subtitle='Signal feature only, KNN, K=15,weights="distance"',n_estimators=15,
                                                        )
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\signal_feature_knn_results.png',dpi=500)
fig=eegPipelineFunctions.visualize_auc_precision_recall(graph_features_dict,keys,clf_='knn',
                                                        subtitle='Graph feature only, KNN, K=15,weights="distance"',n_estimators=15,)
fig.tight_layout(pad=4.5)
fig.savefig(file_dir+'results\\graph_feature_knn_results.png',dpi=500)














