# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:14:44 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
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
#    file_dir = 'D:\\NING - spindle\\training set\\road_trip_more_channels\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
#    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip_more_channels\\'
    os.chdir(file_dir)
################################### Random forest #################################    
signal_features_indivisual_results,graph_features_indivisual_results,combine_features_indivisual_results={},{},{}
signal_features_indivisual_results = eegPipelineFunctions.cross_validation_report(signal_features_indivisual_results,0,
                                                                                  clf_='RF',file_dir=file_dir,compute='signal')
graph_features_indivisual_results = eegPipelineFunctions.cross_validation_report(graph_features_indivisual_results,0,
                                                                                 clf_='RF',file_dir=file_dir,compute='graph')
combine_features_indivisual_results = eegPipelineFunctions.cross_validation_report(combine_features_indivisual_results,0,
                                                                                   clf_='RF',file_dir=file_dir,compute='combine')
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_RF.csv',index=False)
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_RF.csv',index=False)  
combine_features_indivisual_results.to_csv(file_dir+'individual_combine_feature_RF.csv',index=False)
################################### support vector machine ###########################
signal_features_indivisual_results,graph_features_indivisual_results,combine_features_indivisual_results={},{},{}
signal_features_indivisual_results = eegPipelineFunctions.cross_validation_report(signal_features_indivisual_results,0,
                                                                                  clf_='svm',file_dir=file_dir,compute='signal')
graph_features_indivisual_results = eegPipelineFunctions.cross_validation_report(graph_features_indivisual_results,0,
                                                                                 clf_='svm',file_dir=file_dir,compute='graph')
combine_features_indivisual_results = eegPipelineFunctions.cross_validation_report(combine_features_indivisual_results,0,
                                                                                   clf_='svm',file_dir=file_dir,compute='combine')
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_svm.csv',index=False)
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_svm.csv',index=False)  
combine_features_indivisual_results.to_csv(file_dir+'individual_combine_feature_svm.csv',index=False)
################################## logistic regression ##################################################
signal_features_indivisual_results,graph_features_indivisual_results,combine_features_indivisual_results={},{},{}
signal_features_indivisual_results = eegPipelineFunctions.cross_validation_report(signal_features_indivisual_results,0,
                                                                                  clf_='logistic',file_dir=file_dir,compute='signal')
graph_features_indivisual_results = eegPipelineFunctions.cross_validation_report(graph_features_indivisual_results,0,
                                                                                 clf_='logistic',file_dir=file_dir,compute='graph')
combine_features_indivisual_results = eegPipelineFunctions.cross_validation_report(combine_features_indivisual_results,0,
                                                                                   clf_='logistic',file_dir=file_dir,compute='combine')
signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_logistic.csv',index=False)
graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_logistic.csv',index=False)  
combine_features_indivisual_results.to_csv(file_dir+'individual_combine_feature_logistic.csv',index=False)

################################ TPOT ############################################      
#from sklearn.pipeline import make_pipeline, make_union
#from sklearn.decomposition import PCA
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import FunctionTransformer,Normalizer
#from sklearn.naive_bayes import BernoulliNB,GaussianNB
#from copy import copy
#from sklearn.svm import LinearSVC
#from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
#from sklearn.linear_model import LogisticRegression   
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.feature_selection import VarianceThreshold,SelectPercentile, f_classif     
#clfs_graph = {1.5:make_pipeline(
#    make_union(VotingClassifier([("est", GradientBoostingClassifier(max_depth=1, 
#                                                                    max_features=0.2, 
#                                                                    min_samples_leaf=5, 
#                                                                    min_samples_split=2, 
#                                                                    n_estimators=100, 
#                                                                    subsample=0.45))]), 
#                                                FunctionTransformer(copy)),
#                                                LogisticRegression(C=0.5)),
#        2.0:LogisticRegression(C=0.1, dual=False),
#        2.5:LinearSVC(C=0.001, loss="hinge", penalty="l2", tol=0.1),
#        3.0:make_pipeline(make_union(
#                                    Normalizer(norm="max"),
#                                    FunctionTransformer(copy)),
#                                    KNeighborsClassifier(n_neighbors=95, p=1)),
#        3.5:LogisticRegression(C=5.0),
#        4.0:make_pipeline(make_union(VotingClassifier([("est", BernoulliNB(alpha=100.0, 
#                                                                                          fit_prior=True))]), 
#                                                FunctionTransformer(copy)),
#                                                LogisticRegression(C=0.5, penalty="l2")),
#        4.5:LogisticRegression(),
#        5.0:LogisticRegression(C=0.5, dual=False, penalty="l2")}
#
#clfs_signal = {1.5:make_pipeline(VarianceThreshold(threshold=0.5),
#                                                DecisionTreeClassifier(criterion="entropy", 
#                                                                       max_depth=3, 
#                                                                       min_samples_leaf=8, 
#                                                                       min_samples_split=6)),
#        2.0:make_pipeline(make_union(VotingClassifier([("est", GaussianNB())]), FunctionTransformer(copy)),
#                                                                     LinearSVC(C=15.0, loss="hinge", penalty="l2", tol=0.01)),
#        2.5:LogisticRegression(C=0.01, dual=True),
#        3.0:LinearSVC(dual=True, loss="hinge", penalty="l2", tol=0.001),
#        3.5:LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2", tol=0.1),
#        4.0:make_pipeline(VarianceThreshold(threshold=0.9),
#                                         make_union(VotingClassifier([("est", GaussianNB())]), 
#                                                                     FunctionTransformer(copy)),
#                                                                     KNeighborsClassifier(n_neighbors=100, p=1)),
#        4.5:make_pipeline(SelectPercentile(score_func=f_classif, percentile=33),
#                                         LogisticRegression(C=20.0, dual=False)),
#        5.0:make_pipeline(PCA(iterated_power=7, svd_solver="randomized"),GaussianNB())}        
#        
#signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[],
#                      'matthews_corrcoef_mean':[],
#                      'matthews_corrcoef_std':[]}
#graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[],
#                      'matthews_corrcoef_mean':[],
#                      'matthews_corrcoef_std':[]}
#combine_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[],
#                      'matthews_corrcoef_mean':[],
#                      'matthews_corrcoef_std':[]}
#for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
#    sub_dir = file_dir + directory_1 + '\\'
#    epoch_length = directory_1.split(' ')[1]
#    os.chdir(sub_dir)
#    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
#    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
#    for sub_fold in os.listdir(sub_dir):
#        sub_fold_dir = sub_dir + sub_fold + '\\'
#        os.chdir(sub_fold_dir)
#        sub = sub_fold[:-4]
#        day = sub_fold[4:][-4:]
#        print(sub,day,epoch_length)
#        
#        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
#        #df_cc.append(cc_features)
#        #df_pli.append(pli_features)
#        #df_plv.append(plv_features)
#        label = cc_features['label']
#        cc_features = eegPipelineFunctions.get_real_part(cc_features)
#        pli_features = eegPipelineFunctions.get_real_part(pli_features)
#        plv_features = eegPipelineFunctions.get_real_part(plv_features)
#        cc_features.columns = ['cc_'+name for name in cc_features]
#        pli_features.columns = ['pli_'+name for name in pli_features]
#        plv_features.columns = ['plv_'+name for name in plv_features]
#        cc_features = cc_features.drop('cc_label',1)
#        pli_features = pli_features.drop('pli_label',1)
#        plv_features = plv_features.drop('plv_label',1)
#        df_combine = pd.concat([cc_features,pli_features,plv_features],axis=1)
#        df_combine['label']=label
#        df_two = pd.concat([cc_features, pli_features, plv_features, signal_features],axis=1)
#        try:
#            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_=clfs_signal[float(epoch_length)])
#            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_=clfs_graph[float(epoch_length)])
#            two_temp = eegPipelineFunctions.cross_validation_with_clfs(df_two,clf_=clfs_graph[float(epoch_length)])
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores,MCC=signal_temp
#            signal_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            signal_features_indivisual_results['fpr'].append(fpr)
#            signal_features_indivisual_results['tpr'].append(tpr)
#            signal_features_indivisual_results['precision'].append(precision)
#            signal_features_indivisual_results['recall'].append(recall)
#            signal_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            signal_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            signal_features_indivisual_results['matthews_corrcoef_mean'].append(np.nanmean(MCC))
#            signal_features_indivisual_results['matthews_corrcoef_std'].append(np.nanstd(MCC))
#            signal_features_indivisual_results['subject'].append(sub)
#            signal_features_indivisual_results['day'].append(int(day[-1]))
#            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores,MCC=graph_temp
#            graph_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            graph_features_indivisual_results['fpr'].append(fpr)
#            graph_features_indivisual_results['tpr'].append(tpr)
#            graph_features_indivisual_results['precision'].append(precision)
#            graph_features_indivisual_results['recall'].append(recall)
#            graph_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            graph_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            graph_features_indivisual_results['matthews_corrcoef_mean'].append(np.nanmean(MCC))
#            graph_features_indivisual_results['matthews_corrcoef_std'].append(np.nanstd(MCC))
#            graph_features_indivisual_results['subject'].append(sub)
#            graph_features_indivisual_results['day'].append(int(day[-1]))
#            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores,MCC=two_temp
#            combine_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            combine_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            combine_features_indivisual_results['fpr'].append(fpr)
#            combine_features_indivisual_results['tpr'].append(tpr)
#            combine_features_indivisual_results['precision'].append(precision)
#            combine_features_indivisual_results['recall'].append(recall)
#            combine_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            combine_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            combine_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            combine_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            combine_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            combine_features_indivisual_results['matthews_corrcoef_mean'].append(np.nanmean(MCC))
#            combine_features_indivisual_results['matthews_corrcoef_std'].append(np.nanstd(MCC))
#            combine_features_indivisual_results['subject'].append(sub)
#            combine_features_indivisual_results['day'].append(int(day[-1]))
#            combine_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#        except:
#            print(sub_fold,Counter(label),'not enough samples')
#signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
#graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results) 
#combine_features_indivisual_results = pd.DataFrame(combine_features_indivisual_results)       
#signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_TPOT.csv',index=False)
#graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_TPOT.csv',index=False)         
#combine_features_indivisual_results.to_csv(file_dir+'individual_combine_feature_TPOT.csv',index=False)  




#signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
#    sub_dir = file_dir + directory_1 + '\\'
#    epoch_length = directory_1.split(' ')[1]
#    os.chdir(sub_dir)
#    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
#    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
#    for sub_fold in os.listdir(sub_dir):
#        sub_fold_dir = sub_dir + sub_fold + '\\'
#        os.chdir(sub_fold_dir)
#        sub = sub_fold[:-4]
#        day = sub_fold[4:][-4:]
#        print(sub,day,epoch_length)
#        
#        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
#        #df_cc.append(cc_features)
#        #df_pli.append(pli_features)
#        #df_plv.append(plv_features)
#        label = cc_features['label']
#        cc_features = eegPipelineFunctions.get_real_part(cc_features)
#        pli_features = eegPipelineFunctions.get_real_part(pli_features)
#        plv_features = eegPipelineFunctions.get_real_part(plv_features)
#        cc_features.columns = ['cc_'+name for name in cc_features]
#        pli_features.columns = ['pli_'+name for name in pli_features]
#        plv_features.columns = ['plv_'+name for name in plv_features]
#        cc_features = cc_features.drop('cc_label',1)
#        pli_features = pli_features.drop('pli_label',1)
#        plv_features = plv_features.drop('plv_label',1)
#        df_combine = pd.concat([cc_features,pli_features,plv_features],axis=1)
#        df_combine['label']=label
#        try:
#            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='logistic')
#            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='logistic')
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=signal_temp
#            signal_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            signal_features_indivisual_results['fpr'].append(fpr)
#            signal_features_indivisual_results['tpr'].append(tpr)
#            signal_features_indivisual_results['precision'].append(precision)
#            signal_features_indivisual_results['recall'].append(recall)
#            signal_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            signal_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            signal_features_indivisual_results['subject'].append(sub)
#            signal_features_indivisual_results['day'].append(int(day[-1]))
#            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
#            graph_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            graph_features_indivisual_results['fpr'].append(fpr)
#            graph_features_indivisual_results['tpr'].append(tpr)
#            graph_features_indivisual_results['precision'].append(precision)
#            graph_features_indivisual_results['recall'].append(recall)
#            graph_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            graph_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            graph_features_indivisual_results['subject'].append(sub)
#            graph_features_indivisual_results['day'].append(int(day[-1]))
#            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#        except:
#            print(sub_fold,Counter(label),'not enough samples')
#signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
#graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)
#signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_regression.csv',index=False)
#graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_regression.csv',index=False)       
#        
#    
#signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#combine_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
#    sub_dir = file_dir + directory_1 + '\\'
#    epoch_length = directory_1.split(' ')[1]
#    os.chdir(sub_dir)
#    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
#    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
#    for sub_fold in os.listdir(sub_dir):
#        sub_fold_dir = sub_dir + sub_fold + '\\'
#        os.chdir(sub_fold_dir)
#        sub = sub_fold[:-4]
#        day = sub_fold[4:][-4:]
#        print(sub,day,epoch_length)
#        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
#        #df_cc.append(cc_features)
#        #df_pli.append(pli_features)
#        #df_plv.append(plv_features)
#        label = cc_features['label']
#        cc_features = eegPipelineFunctions.get_real_part(cc_features)
#        pli_features = eegPipelineFunctions.get_real_part(pli_features)
#        plv_features = eegPipelineFunctions.get_real_part(plv_features)
#        cc_features.columns = ['cc_'+name for name in cc_features]
#        pli_features.columns = ['pli_'+name for name in pli_features]
#        plv_features.columns = ['plv_'+name for name in plv_features]
#        cc_features = cc_features.drop('cc_label',1)
#        pli_features = pli_features.drop('pli_label',1)
#        plv_features = plv_features.drop('plv_label',1)
#        df_combine = pd.concat([cc_features,pli_features,plv_features],axis=1)
#        df_combine['label']=label
#        df_two = pd.concat([cc_features, pli_features, plv_features, signal_features],axis=1)
#        try:
#            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='RF')
#            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='RF')
#            two_temp = eegPipelineFunctions.cross_validation_with_clfs(df_two,clf_='RF')
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores,MCC=signal_temp
#            signal_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            signal_features_indivisual_results['fpr'].append(fpr)
#            signal_features_indivisual_results['tpr'].append(tpr)
#            signal_features_indivisual_results['precision'].append(precision)
#            signal_features_indivisual_results['recall'].append(recall)
#            signal_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            signal_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            signal_features_indivisual_results['subject'].append(sub)
#            signal_features_indivisual_results['day'].append(int(day[-1]))
#            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(MCC),np.std(MCC)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
#            graph_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            graph_features_indivisual_results['fpr'].append(fpr)
#            graph_features_indivisual_results['tpr'].append(tpr)
#            graph_features_indivisual_results['precision'].append(precision)
#            graph_features_indivisual_results['recall'].append(recall)
#            graph_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            graph_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            graph_features_indivisual_results['subject'].append(sub)
#            graph_features_indivisual_results['day'].append(int(day[-1]))
#            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=two_temp
#            combine_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            combine_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            combine_features_indivisual_results['fpr'].append(fpr)
#            combine_features_indivisual_results['tpr'].append(tpr)
#            combine_features_indivisual_results['precision'].append(precision)
#            combine_features_indivisual_results['recall'].append(recall)
#            combine_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            combine_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            combine_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            combine_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            combine_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            combine_features_indivisual_results['subject'].append(sub)
#            combine_features_indivisual_results['day'].append(int(day[-1]))
#            combine_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#        except:
#            print(sub_fold,Counter(label),'not enough samples')
#signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
#graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)            
#signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_RF.csv',index=False)
#graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_RF.csv',index=False)   
##pickle.dump(signal_features_indivisual_results,open(file_dir+'individual_signal_feature_RF.p','wb'))
##pickle.dump(graph_features_indivisual_results,open(file_dir+'individual_graph_feature_RF.p','wb'))
#        
#        
#signal_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#graph_features_indivisual_results = {'subject':[],'day':[],'epoch_length':[],
#                                      'auc_score_mean':[],'auc_score_std':[],
#                                      'fpr':[],'tpr':[],
#                                      'precision':[],'recall':[],
#                                      'precision_mean':[],'precision_std':[],
#                                      'recall_mean':[],'recall_std':[],
#                                      'area_under_precision_recall':[]}
#for directory_1 in [f for f in os.listdir(file_dir) if ('epoch_length' in f)]:
#    sub_dir = file_dir + directory_1 + '\\'
#    epoch_length = directory_1.split(' ')[1]
#    os.chdir(sub_dir)
#    #signal_features_indivisual_results[directory_1],graph_features_indivisual_results[directory_1]={},{}
#    #df_cc, df_pli, df_plv, df_signal,df_graph = [],[],[],[],[]
#    for sub_fold in os.listdir(sub_dir):
#        sub_fold_dir = sub_dir + sub_fold + '\\'
#        os.chdir(sub_fold_dir)
#        sub = sub_fold[:-4]
#        day = sub_fold[4:][-4:]
#        print(sub,day,epoch_length)
#        
#        cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
#        #df_cc.append(cc_features)
#        #df_pli.append(pli_features)
#        #df_plv.append(plv_features)
#        label = cc_features['label']
#        cc_features = eegPipelineFunctions.get_real_part(cc_features)
#        pli_features = eegPipelineFunctions.get_real_part(pli_features)
#        plv_features = eegPipelineFunctions.get_real_part(plv_features)
#        cc_features.columns = ['cc_'+name for name in cc_features]
#        pli_features.columns = ['pli_'+name for name in pli_features]
#        plv_features.columns = ['plv_'+name for name in plv_features]
#        cc_features = cc_features.drop('cc_label',1)
#        pli_features = pli_features.drop('pli_label',1)
#        plv_features = plv_features.drop('plv_label',1)
#        df_combine = pd.concat([cc_features,pli_features,plv_features],axis=1)
#        df_combine['label']=label
#        try:
#            signal_temp = eegPipelineFunctions.cross_validation_with_clfs(signal_features,clf_='svm')
#            graph_temp = eegPipelineFunctions.cross_validation_with_clfs(df_combine,clf_='svm')
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=signal_temp
#            signal_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            signal_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            signal_features_indivisual_results['fpr'].append(fpr)
#            signal_features_indivisual_results['tpr'].append(tpr)
#            signal_features_indivisual_results['precision'].append(precision)
#            signal_features_indivisual_results['recall'].append(recall)
#            signal_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            signal_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            signal_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            signal_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            signal_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            signal_features_indivisual_results['subject'].append(sub)
#            signal_features_indivisual_results['day'].append(int(day[-1]))
#            signal_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'signal:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#            auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores=graph_temp
#            graph_features_indivisual_results['auc_score_mean'].append(np.nanmean(auc_score))
#            graph_features_indivisual_results['auc_score_std'].append(np.std(auc_score))
#            graph_features_indivisual_results['fpr'].append(fpr)
#            graph_features_indivisual_results['tpr'].append(tpr)
#            graph_features_indivisual_results['precision'].append(precision)
#            graph_features_indivisual_results['recall'].append(recall)
#            graph_features_indivisual_results['precision_mean'].append(np.nanmean(precision_scores))
#            graph_features_indivisual_results['precision_std'].append(np.std(precision_scores))
#            graph_features_indivisual_results['recall_mean'].append(np.nanmean(recall_scores))
#            graph_features_indivisual_results['recall_std'].append(np.std(recall_scores))
#            graph_features_indivisual_results['area_under_precision_recall'].append(average_scores)
#            graph_features_indivisual_results['subject'].append(sub)
#            graph_features_indivisual_results['day'].append(int(day[-1]))
#            graph_features_indivisual_results['epoch_length'].append(float(epoch_length))
#            print(sub_fold,Counter(label),'graph:%.2f +/-%.2f'%(np.nanmean(auc_score),np.std(auc_score)))
#        except:
#            print(sub_fold,Counter(label),'not enough samples')
#signal_features_indivisual_results = pd.DataFrame(signal_features_indivisual_results)
#graph_features_indivisual_results = pd.DataFrame(graph_features_indivisual_results)
#signal_features_indivisual_results.to_csv(file_dir+'individual_signal_feature_svm.csv',index=False)
#graph_features_indivisual_results.to_csv(file_dir+'individual_graph_feature_svm.csv',index=False)          
