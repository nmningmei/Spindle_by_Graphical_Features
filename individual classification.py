# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:14:44 2017

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

signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1.split(' ')[1]
    os.chdir(sub_dir)
    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        sub = sub_fold[:-4]
        day = sub_fold[4:][-4:]
        print(sub,day,epoch_length)
        
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
            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='logistic')
            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='logistic')
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=signal_temp
            signal_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            signal_features_indivisual_results['fpr'].append(fpr)
            signal_features_indivisual_results['tpr'].append(tpr)
            signal_features_indivisual_results['precision'].append(precision)
            signal_features_indivisual_results['recall'].append(recall)
            signal_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            signal_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            signal_features_indivisual_results['subject'].append(sub)
            signal_features_indivisual_results['day'].append(int(day[-1]))
            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
            graph_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            graph_features_indivisual_results['fpr'].append(fpr)
            graph_features_indivisual_results['tpr'].append(tpr)
            graph_features_indivisual_results['precision'].append(precision)
            graph_features_indivisual_results['recall'].append(recall)
            graph_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            graph_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            graph_features_indivisual_results['subject'].append(sub)
            graph_features_indivisual_results['day'].append(int(day[-1]))
            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
        except:
            print(sub_fold,Counter(label),'not enough samples')
signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_regression.csv')
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_regression.csv')       
        
        
signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1.split(' ')[1]
    os.chdir(sub_dir)
    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        sub = sub_fold[:-4]
        day = sub_fold[4:][-4:]
        print(sub,day,epoch_length)
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
            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='RF')
            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='RF')
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=signal_temp
            signal_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            signal_features_indivisual_results['fpr'].append(fpr)
            signal_features_indivisual_results['tpr'].append(tpr)
            signal_features_indivisual_results['precision'].append(precision)
            signal_features_indivisual_results['recall'].append(recall)
            signal_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            signal_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            signal_features_indivisual_results['subject'].append(sub)
            signal_features_indivisual_results['day'].append(int(day[-1]))
            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
            graph_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            graph_features_indivisual_results['fpr'].append(fpr)
            graph_features_indivisual_results['tpr'].append(tpr)
            graph_features_indivisual_results['precision'].append(precision)
            graph_features_indivisual_results['recall'].append(recall)
            graph_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            graph_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            graph_features_indivisual_results['subject'].append(sub)
            graph_features_indivisual_results['day'].append(int(day[-1]))
            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
        except:
            print(sub_fold,Counter(label),'not enough samples')
signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)            
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_RF.csv')
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_RF.csv')   
#pickle.dump(signal_features_indivisual_results,open(file_dir+'individual_signal_feature_RF.p','wb'))
#pickle.dump(graph_features_indivisual_results,open(file_dir+'individual_graph_feature_RF.p','wb'))
        
        
signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
                                      'auc_score_mean':[],'auc_score_std':[],
                                      'fpr':[],'tpr':[],
                                      'precision':[],'recall':[],
                                      'precision_mean':[],'precision_std':[],
                                      'recall_mean':[],'recall_std':[],
                                      'area_under_precision_recall':[]}
for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
    sub_dir = file_dir + directory_1 + '\\'
    epoch_length = directory_1.split(' ')[1]
    os.chdir(sub_dir)
    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        sub = sub_fold[:-4]
        day = sub_fold[4:][-4:]
        print(sub,day,epoch_length)
        
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
            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='svm')
            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='svm')
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=signal_temp
            signal_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            signal_features_indivisual_results['fpr'].append(fpr)
            signal_features_indivisual_results['tpr'].append(tpr)
            signal_features_indivisual_results['precision'].append(precision)
            signal_features_indivisual_results['recall'].append(recall)
            signal_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            signal_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            signal_features_indivisual_results['subject'].append(sub)
            signal_features_indivisual_results['day'].append(int(day[-1]))
            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
            graph_features_indivisual_results['auc_score_mean'].append(np.mean(auc_score))
            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
            graph_features_indivisual_results['fpr'].append(fpr)
            graph_features_indivisual_results['tpr'].append(tpr)
            graph_features_indivisual_results['precision'].append(precision)
            graph_features_indivisual_results['recall'].append(recall)
            graph_features_indivisual_results['precision_mean'].append(np.mean(precision_scores))
            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
            graph_features_indivisual_results['recall_mean'].append(np.mean(recall_scores))
            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
            graph_features_indivisual_results['subject'].append(sub)
            graph_features_indivisual_results['day'].append(int(day[-1]))
            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
        except:
            print(sub_fold,Counter(label),'not enough samples')
signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_svm.csv')
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_svm.csv')          
        
        
        
        
        
        