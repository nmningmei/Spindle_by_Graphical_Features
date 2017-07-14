# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:22:46 2017

@author: ning
"""
import mne
import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import laplacian
import os
import networkx as nx
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,precision_recall_curve,auc,precision_score,recall_score,average_precision_score,classification_report,matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from time import sleep
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    return complex_phase_diff
def phase_lag_index(data1, data2):
    PLI = np.angle(data1) - np.angle(data2)
    PLI[PLI<-np.pi] += 2*np.pi
    PLI[PLI>-np.pi] -= 2*np.pi
    return PLI

def spindle_check(x):
    import re
    if re.compile('spindle',re.IGNORECASE).search(x):
        return True
    else:
        return False
def get_data_ready(filename,channelList,annotation_file,l_freq=11,h_freq=16,epoch_length=5,overalapping=0.2,
                   ):
    raw = mne.io.read_raw_fif(filename,preload=True)
    if channelList is not None:
        raw.pick_channels(channelList)
    else:
        raw.drop_channels(['LOc','ROc'])
    raw.filter(l_freq,h_freq)
    a = np.arange(0,raw.times[-1],epoch_length - epoch_length*overalapping)
    events = np.array([a*raw.info['sfreq'],[int(epoch_length)]*len(a),[int(1)*raw.info['sfreq']]*len(a)],dtype=int).T
    epochs = mne.Epochs(raw,events,tmin=0,tmax=epoch_length,baseline=None,preload=True,proj=False)
    print('down sample to 100 Hz ........................')
    epochs.resample(100)
    print('find target event annotations')
    annotation = pd.read_csv(annotation_file)
    spindles = annotation[annotation['Annotation'].apply(spindle_check)]
    print('number of spindles marked: %d' %(len(spindles)))
    manual_label,temp = discritized_onset_label_manual(epochs,spindles,epoch_length,front=0,back=0)
    return epochs,manual_label,temp

def featureExtraction(epochs):
    features = ['mean','variance','delta_mean',
          'delta_variance','change_variance',
         'activity','mobility','complexity','skewness_of_amplitude_spectrum',
         'spectral_entropy']
    epochFeatures = {name:[] for name in features}
    for ii, epoch_data in enumerate(epochs):
        epoch_data  = epoch_data[:,:-1].T
        #print('computing features for epoch %d' %(ii+1))
        epochFeatures['mean'].append(np.mean(epoch_data))
        epochFeatures['variance'].append(np.var(epoch_data))
        startRange = epoch_data[:-1,:]
        endRange = epoch_data[1:]
        epochFeatures['delta_mean'].append(np.mean(endRange - startRange))
        epochFeatures['delta_variance'].append(np.mean(np.var(endRange - startRange,axis=0)))
        if ii == 0:
            epochFeatures['change_variance'].append(0)
        elif ii == 1:
            epochFeatures['change_variance'].append(np.mean(np.var(epoch_data - epochFeatures['mean'][ii-1],axis=0)))
        else:
            epochFeatures['change_variance'].append(np.mean(np.var(epoch_data - epochFeatures['mean'][ii-1] - epochFeatures['mean'][ii-2],axis=0)))
        
        activity = np.var(epoch_data)
        epochFeatures['activity'].append(activity)
        tempData = startRange - endRange
        diff1 = np.std(tempData)
        mobility = np.std(tempData)/np.sqrt(activity)
        epochFeatures['mobility'].append(mobility)
        
        startRange = epoch_data[:-2,:]
        endRange = epoch_data[2:,:]
        tempData = endRange - startRange
        complexity = (np.std(tempData)/diff1)/(diff1/np.sqrt(activity))
        epochFeatures['complexity'].append(complexity)
        
        specEnt = np.zeros(epoch_data.shape[1])
        skAmp = np.zeros(epoch_data.shape[1])
        for ii in range(len(specEnt)):
            this_epoch = epoch_data[:,ii]
            ampSpec = abs(np.fft.fft(this_epoch))
            skAmp[ii] = stats.skew(ampSpec)
            ampSpec  /= sum(ampSpec)
            specEnt[ii] = -sum(ampSpec * np.log2(ampSpec))
        skAmp[np.isnan(skAmp)] = 0;skAmp = np.mean(skAmp)
        specEnt[np.isnan(specEnt)] = 0 ; specEnt = np.mean(specEnt)
        epochFeatures['spectral_entropy'].append(specEnt)
        epochFeatures['skewness_of_amplitude_spectrum'].append(skAmp)
    return epochFeatures
def connectivity(epochs):       
    ch_names = epochs.ch_names
    connectivity=[]
    for ii, epoch_data in enumerate(epochs):
        epoch_data  = epoch_data[:,:-1].T
        #print('computing connectivity for epoch %d' %(ii+1))
        dist_list_plv = np.zeros(shape=(len(ch_names),len(ch_names)))
        dist_list_pli = np.zeros(shape=(len(ch_names),len(ch_names)))
        for node_1 in range(len(ch_names)):
            for node_2 in range(len(ch_names)):
                if node_1 != node_2:
                    data_1 = epoch_data[node_1,:]
                    data_2 = epoch_data[node_2,:]
                    PLV=phase_locking_value(np.angle(signal.hilbert(data_1,axis=0)),
                                             np.angle(signal.hilbert(data_2,axis=0)))
                    dist_list_plv[node_1,node_2]=np.abs(np.mean(PLV))
                    PLI=np.angle(signal.hilbert(data_1,axis=0))-np.angle(signal.hilbert(data_2,axis=0))
                    dist_list_pli[node_1,node_2]=np.abs(np.mean(np.sign(PLI)))
        dist_list_cc = squareform(pdist(epoch_data.T,'correlation'))
        dist_list_cc = abs(1 - dist_list_cc)
        np.fill_diagonal(dist_list_cc,0)
        connectivity.append([dist_list_plv,dist_list_pli,dist_list_cc])
    return connectivity
def thresholding(threshold, attribute):
    adjacency = []
    for ii,attr in enumerate(attribute):
        adjacency.append(np.array(attr > threshold,dtype=int))
    return adjacency
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None
def extractGraphFeatures(adjacency):
    features = ['average_degree','clustering_coefficient','eccentricity','diameter','radius','path_length',
                'central_point','number_edge','spectral_radius','second_spectral_radius',
                'adjacency_trace','adjacency_energy','spectral_gap','lap_trace','lap_energy',
                'lap_zero','lap_one','lap_two','lap_trace_normal']
    results = {name:[] for name in features}
    for ii,a in enumerate(adjacency):
        
        G = nx.from_numpy_matrix(a)
        if nx.is_connected(G):
            #print('computing connected graphic features of epoch %d'%(ii+1))
            average_degree = nx.average_neighbor_degree(G)
            average_degree = np.mean([v for v in average_degree.values()])
            
            clustering_coefficient = nx.average_clustering(G)
            
            eccentricity = nx.eccentricity(G)
            average_eccentricity = np.mean([v for v in eccentricity.values()])
            diameter = nx.diameter(G)
            radius = nx.radius(G)
            Path_length=[]
            for j in range(6):
                for k in range(6):
                    if j != k:
                        Path_length.append(nx.dijkstra_path_length(G,j,k))
            average_path_length=np.mean(Path_length)
            
#            connect_component_ratio = None
#            number_connect_components = None
#            average_component_size = None
#            isolated_point = None
#            end_point = None
            central_point = (np.array([v for v in eccentricity.values()]) == radius).astype(int)
            central_point = central_point.sum() / central_point.shape[0]
            
            number_edge = nx.number_of_edges(G)
            
            spectral_radius = max(np.linalg.eigvals(a))
            second_spectral_radius = second_largest(np.linalg.eigvals(a))
            adjacency_trace = np.linalg.eigvals(a).sum()
            adjacency_energy = np.sum(np.linalg.eigvals(a)**2)
            spectral_gap = spectral_radius- second_spectral_radius
            
            Laplacian_M_unnormal = laplacian(a,normed=False,)
            laplacian_trace = np.linalg.eigvals(Laplacian_M_unnormal).sum()
            laplacian_energy = np.sum(np.linalg.eigvals(Laplacian_M_unnormal)**2)
            Laplacian_M_normal = laplacian(a,normed=True)
            laplacian_zero = len(Laplacian_M_normal == 0)
            laplacian_one  = len(Laplacian_M_normal == 1)
            laplacian_two  = len(Laplacian_M_normal == 2)
            laplacian_trace_normal = np.linalg.eigvals(Laplacian_M_normal).sum()
            
            results['average_degree'].append(average_degree)
            results['clustering_coefficient'].append(clustering_coefficient)
            results['eccentricity'].append(average_eccentricity)
            results['diameter'].append(diameter)
            results['radius'].append(radius)
            results['path_length'].append(average_path_length)
            results['central_point'].append(central_point)
            results['number_edge'].append(number_edge)
            results['spectral_radius'].append(spectral_radius)
            results['second_spectral_radius'].append(second_spectral_radius)
            results['adjacency_trace'].append(adjacency_trace)
            results['adjacency_energy'].append(adjacency_energy)
            results['spectral_gap'].append(spectral_gap)
            results['lap_trace'].append(laplacian_trace)
            results['lap_energy'].append(laplacian_energy)
            results['lap_zero'].append(laplacian_zero)
            results['lap_one'].append(laplacian_one)
            results['lap_two'].append(laplacian_two)
            results['lap_trace_normal'].append(laplacian_trace_normal)
        else:
            #print('computing disconnected graphic features of epoch %d'%(ii+1))
            for name in results.keys():
                results[name].append(-99)
        
    results = pd.DataFrame(results)
    return results
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix:
        spindle_start = spindle - 0.5
        spindle_end   = spindle + 1.5
        a =  np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a        
def discritized_onset_label_manual(epochs,df,spindle_segment_length,front=300,back=100):
    start_times = epochs.events[:,0]/(np.unique(epochs.events[:,-1])/100)
    end_times = start_times + epochs.events[:,1] * 100
    discritized_time_intervals = np.vstack((start_times,end_times)).T
    discritized_time_intervals = discritized_time_intervals / 100
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    temp=[]
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            temp.append([time_interval,spindle])
            if spindle_comparison(time_interval,spindle,spindle_segment_length):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,temp
def get_real_part(df):
    temp = {}
    for name in df.columns:
        try:
            temp[name] = pd.to_numeric(df[name])
        except:
            a = np.array([np.real(np.complex(value)) for value in df[name].values])
            temp[name] = a
    return pd.DataFrame(temp)
#def cross_validation_pipeline(dfs,cv=None):
#    data = dfs.values   
#    X, Y = data[:,:-1], data[:,-1]
#    if cv == None:
#        cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=np.random.randint(10000,20000))
#    else:
#        cv = KFold(n_splits=cv,shuffle=True,random_state=12334)
#    results = []
#    for train, test in cv.split(X,Y):
#        clf = Pipeline([('scaler',StandardScaler()),
#                        ('estimator',LogisticRegressionCV(Cs=np.logspace(-3,3,7),
#                          max_iter=int(1e4),
#                          tol=1e-4,
#                          scoring='roc_auc',solver='sag',cv=10,
#                          class_weight={1:np.count_nonzero(Y)/len(Y),0:1-(np.count_nonzero(Y)/len(Y))}))])
#        clf.fit(X[train],Y[train])
#        fpr,tpr,_ = roc_curve(Y[test],clf.predict_proba(X[test])[:,-1])
#        auc_score = auc(fpr,tpr)
#        precision,recall,_ = precision_recall_curve(Y[test],clf.decision_function(X[test]))
#        precision_scores = precision_score(Y[test],clf.predict(X[test]), average='micro')
#        recall_scores    = recall_score(Y[test],clf.predict(X[test]), average='micro')
#        average_scores = average_precision_score(Y[test],clf.predict(X[test]))
#        results.append([auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores])
#    return results
def RF_cv(clf,X,Y,train,test,ratio):
    clf.fit(X[train],Y[train])
    fpr,tpr,_ = roc_curve(Y[test],clf.predict_proba(X[test])[:,-1])
    auc_score = auc(fpr,tpr)
    try:
        precision,recall,_ = precision_recall_curve(Y[test],clf.predict_proba(X[test])[:,-1])
        #print(Y[test],clf.predict(X[test]))
        precision_scores = precision_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio, average='binary')
        recall_scores    = recall_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio, average='binary')
        average_scores = average_precision_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        MCC = matthews_corrcoef(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        confm = confusion_matrix(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        confm = confm / confm.sum(axis=1)[:,np.newaxis]
        print(classification_report(Y[test],clf.predict_proba(X[test])[:,-1]>ratio))
    except:
        precision,recall,_ = precision_recall_curve(Y[test],clf.predict_proba(X[test])[:,-1])
        #print(Y[test],clf.predict(X[test]))
        precision_scores = precision_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio, average='binary')
        recall_scores    = recall_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio, average='binary')
        average_scores = average_precision_score(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        MCC = matthews_corrcoef(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        confm = confusion_matrix(Y[test],clf.predict_proba(X[test])[:,-1]>ratio)
        confm = confm / confm.sum(axis=1)[:,np.newaxis]
        print(classification_report(Y[test],clf.predict_proba(X[test])[:,-1]>ratio))
    return fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm
def SVM_cv(clf,X,Y,train,test):
    clf.fit(X[train],Y[train])
    true = Y[test];predict_prob=clf.decision_function(X[test]);
    fpr,tpr,T = roc_curve(true,predict_prob)
    ratio_ = T.mean()
    predict=np.array(predict_prob>ratio_,dtype=int)
    auc_score = auc(fpr,tpr)
    precision,recall,_ = precision_recall_curve(true,predict_prob)
    precision_scores = precision_score(true,predict,average='binary')
    recall_scores = recall_score(true,predict,average='binary')
    average_scores = average_precision_score(true,predict)
    MCC = matthews_corrcoef(true,predict,)
    confm = confusion_matrix(true,predict)
    confm = confm / confm.sum(axis=1)[:,np.newaxis]
    print(classification_report(true,predict))
    return fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm
def xgb_cv(clf,X,Y, train,test,ratio):
    clf.fit(X[train],Y[train])
    true = Y[test];predic_prob=clf.predict_proba(X[test])[:,-1]
    fpr,tpr,T, = roc_curve(true,predic_prob)
    auc_score = auc(fpr,tpr)
    predict = predic_prob > ratio
    precision,recall,_ = precision_recall_curve(true,predic_prob)
    precision_scores = precision_score(true,predict,average='binary')
    recall_scores = recall_score(true,predict,average='binary')
    average_scores = average_precision_score(true,predict)
    MCC = matthews_corrcoef(true,predict,)
    confm = confusion_matrix(true,predict)
    confm = confm / confm.sum(axis=1)[:,np.newaxis]
    print(classification_report(true,predict))
    return fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm

#parameters = {'weights':('uniform', 'distance'), 'n_neighbors':np.arange(4,16)}
#clf=KNeighborsClassifier()
#from sklearn.model_selection import GridSearchCV
#clf_op = GridSearchCV(clf,parameters,scoring='roc_auc',cv=5)
#clf_op.fit(X,Y)

    
def cross_validation_with_clfs(dfs,clf_ = 'logistic', cv=None,kernel='rbf',weights=5,n_estimators=50,C = 1.):
    from collections import Counter
    print('cross validation %s'%clf_)
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    if cv is None:
        cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=12345)
    elif (type(cv) is int) or (type(cv) is float):
        cv = StratifiedKFold(n_splits=int(cv),shuffle=True,random_state=12345)
    else:
        cv = KFold(n_splits=cv,shuffle=True,random_state=12334)
    auc_score_,fpr_,tpr_,precision_,recall_,precision_scores_,recall_scores_,average_scores_,MCC_,confM_=[],[],[],[],[],[],[],[],[],[]
    
    for jj,(train, test) in enumerate(cv.split(X,Y)):
        print('cv %d'%(jj+1))
        ratio =  list(Counter(Y[train]).values())[1]/(list(Counter(Y[train]).values())[0]+list(Counter(Y[train]).values())[1])
        if clf_ is 'logistic':
#            clf=Pipeline([('scaler',StandardScaler()),
#                            ('estimator',LogisticRegressionCV(Cs=np.logspace(-4,3,8),n_jobs=2,
#                              max_iter=int(1e4),
#                              tol=1e-4,
#                              scoring='roc_auc',solver='sag',cv=5,random_state=12345,#class_weight={1:10,0:1}))])
#                              class_weight={1:weights/(1-ratio)}))])
            clf=Pipeline([('scaler',StandardScaler()),
                          ('estimator',LogisticRegression(C=C,max_iter=int(1e4),
                                                          tol=1e-3,
                                                          random_state=12345,
                                                          class_weight={1:weights/(1-ratio)}))])
            fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm=SVM_cv(clf,X,Y,train,test)
        elif clf_ == 'svm':
            n_estimators=n_estimators
            clf=Pipeline([('scaler',StandardScaler()),
                            ('estimator',SVC(C=C,kernel=kernel,
#                              max_iter=int(1e4),
#                              tol=1e-4,
                              class_weight={1:weights/(1-ratio)},
                              probability=False,random_state=12345))])
#            clf = BaggingClassifier(clf,max_samples=1.0 / n_estimators,n_estimators=n_estimators)
            fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm=SVM_cv(clf,X,Y,train,test)
        elif clf_ == 'RF':
            clf=Pipeline([('scaler',StandardScaler()),
                          ('estimator',RandomForestClassifier(n_estimators=n_estimators,random_state=12345,criterion='gini',#))])
                                                              class_weight={1:weights/(1-ratio)},))])
            fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm=RF_cv(clf,X,Y,train,test,ratio)
        elif clf_ == 'xgb':
            clf=Pipeline([('scaler',StandardScaler()),
                          ('estimator',XGBClassifier())])
            fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm=xgb_cv(clf,X,Y,train,test,ratio)
        elif clf_== 'knn':
            clf=Pipeline([('scaler',StandardScaler()),
                          ('estimator',KNeighborsClassifier(n_neighbors=n_estimators,weights='distance'))])
            fpr, tpr, auc_score,precision, recall,average_scores, precision_scores,recall_scores,MCC,confm=xgb_cv(clf,X,Y,train,test,ratio)
        else:
            clf = clf_
        #sleep(1)
        auc_score_.append(auc_score)
        fpr_.append(fpr)
        tpr_.append(tpr)
        precision_.append(precision)
        recall_.append(recall)
        precision_scores_.append(precision_scores)
        recall_scores_.append(recall_scores)
        average_scores_.append(average_scores)
        MCC_.append(MCC)
        confM_.append(confm)
        
    return auc_score_,fpr_,tpr_,precision_,recall_,precision_scores_,recall_scores_,average_scores_,MCC_,confM_
from scipy import interp
from scipy import interpolate
def interpolate_AUC_precision_recall_curves(fpr, tpr,curve_type='auc'):
    if curve_type == 'auc':
        base_fpr = np.linspace(0,1,101)
        tprs = []
        for fpr_temp, tpr_temp in zip(fpr, tpr):
            tpr_interp = interp(base_fpr,fpr_temp,tpr_temp)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        return base_fpr, mean_tprs,std_tprs,tprs_lower,tprs_upper
    elif curve_type == 'p_r':
        base_fpr = np.linspace(1,0,101)
        tprs = []
        for fpr_temp, tpr_temp in zip(fpr, tpr):
            inter_function = interpolate.interp1d(fpr_temp,tpr_temp)
            tpr_interp = inter_function(base_fpr)
            
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        return base_fpr, mean_tprs,std_tprs,tprs_lower,tprs_upper
def visualize_auc_precision_recall(feature_dictionary,keys,subtitle='',clf_=None,kernel='rbf',weights=5,C =4.,n_estimators=50):
    fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(25,20))
    for ii,(key, dfs, ax) in enumerate(zip(keys,feature_dictionary.values(),axes.flatten())):
        results = cross_validation_with_clfs(dfs,clf_=clf_,kernel=kernel,weights=weights,C=C,n_estimators=n_estimators)
        auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores, MCC, confM = results
        base_fpr, mean_tprs,std_tprs,tprs_lower,tprs_upper = interpolate_AUC_precision_recall_curves(fpr, tpr)
        ax.plot(base_fpr,mean_tprs,color='blue',label='roc auc: %.2f+/-%.2f\n MCC: %.2f+/-%.2f'%(np.mean(auc_score),np.std(auc_score),
                                                                                                 np.mean(MCC),np.std(MCC)))
        ax.fill_between(base_fpr,tprs_lower, tprs_upper, color='blue',alpha=0.3,)
        base_recall, mean_precision, std_precision, precisions_lower, precisions_upper= interpolate_AUC_precision_recall_curves(recall,precision,'p_r')
        ax.plot(base_recall,mean_precision,color='red',
                label='Area under\nprecision-recall curve: %.2f+/-%.2f'%(np.mean(average_scores),
                                                                     np.std(average_scores)))
        ax.fill_between(base_recall,precisions_lower,precisions_upper,color='red',alpha=0.3)
        ax.plot([0, 1], [0, 1], color='navy',  linestyle='--')
        ax.set(xlim=(0,1),ylim=(0,1),title=key,ylabel='True positives (blue)/Precision (red)',
               xlabel='False positives (blue)/Recall (red)')
        ax.legend(loc='best')
        
        print('\n\n'+key+'\n\n', np.mean(confM,axis=0),'\n\n','MCC: %.2f+/-%.2f'%(np.mean(MCC),np.std(MCC)))
        print('roc auc: %.2f+/-%.2f'%(np.mean(auc_score),np.std(auc_score)))
    fig.suptitle(subtitle)
    return fig
from collections import Counter
def cross_validation_report(empty_dictionary, pause_time,clf_='logistic',cv=None,kernel='rbf',file_dir=None,compute='signal',n_estimators=5):
    empty_dictionary={'subject':[],'day':[],'epoch_length':[],
                      'auc_score_mean':[],'auc_score_std':[],
                      'fpr':[],'tpr':[],
                      'precision':[],'recall':[],
                      'precision_mean':[],'precision_std':[],
                      'recall_mean':[],'recall_std':[],
                      'area_under_precision_recall':[],
                      'matthews_corrcoef_mean':[],
                      'matthews_corrcoef_std':[],
                      'confusion_matrix':[]}
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
            cc_features = get_real_part(cc_features)
            pli_features = get_real_part(pli_features)
            plv_features = get_real_part(plv_features)
            cc_features.columns = ['cc_'+name for name in cc_features]
            pli_features.columns = ['pli_'+name for name in pli_features]
            plv_features.columns = ['plv_'+name for name in plv_features]
            cc_features = cc_features.drop('cc_label',1)
            pli_features = pli_features.drop('pli_label',1)
            plv_features = plv_features.drop('plv_label',1)
            df_graph = pd.concat([cc_features,pli_features,plv_features],axis=1)
            df_graph['label']=label
            df_combine = pd.concat([cc_features, pli_features, plv_features, signal_features],axis=1)
            df_work = None
            if compute == 'signal':
                df_work = signal_features
            elif compute == 'graph':
                df_work = df_graph
            elif compute == 'combine':
                df_work = df_combine
            try:
                result_temp = cross_validation_with_clfs(df_work,clf_=clf_,cv=5,weights=10,n_estimators=n_estimators)
                auc_score,fpr,tpr,precision,recall,precision_scores,recall_scores,average_scores,MCC,confM=result_temp
                empty_dictionary['auc_score_mean'].append(np.nanmean(auc_score))
                empty_dictionary['auc_score_std'].append(np.std(auc_score))
                empty_dictionary['fpr'].append(fpr)
                empty_dictionary['tpr'].append(tpr)
                empty_dictionary['precision'].append(precision)
                empty_dictionary['recall'].append(recall)
                empty_dictionary['precision_mean'].append(np.nanmean(precision_scores))
                empty_dictionary['precision_std'].append(np.nanstd(precision_scores))
                empty_dictionary['recall_mean'].append(np.nanmean(recall_scores))
                empty_dictionary['recall_std'].append(np.nanstd(recall_scores))
                empty_dictionary['area_under_precision_recall'].append(average_scores)
                empty_dictionary['matthews_corrcoef_mean'].append(np.nanmean(MCC))
                empty_dictionary['matthews_corrcoef_std'].append(np.nanstd(MCC))
                empty_dictionary['subject'].append(sub)
                empty_dictionary['day'].append(int(day[-1]))
                empty_dictionary['epoch_length'].append(float(epoch_length))
                empty_dictionary['confusion_matrix'].append(confM)
                print(sub_fold,Counter(label),'%s:MCC=%.2f +/-%.2f\nAUC=%.2f+/-%.2f'%(compute, np.nanmean(MCC),np.nanstd(MCC),
                                       np.nanmean(auc_score),np.nanstd(auc_score)))
                sleep(pause_time)
            except:
                print('not enough samples')
    empty_dictionary = pd.DataFrame(empty_dictionary)
    return empty_dictionary
        
                
                
                