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
from scipy.sparse.csgraph import connected_components,laplacian
import os
import networkx as nx

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
    raw.pick_channels(channelList)
    raw.filter(l_freq,h_freq)
    a = np.arange(0,raw.times[-1],epoch_length - epoch_length*overalapping)
    events = np.array([a,[epoch_length]*len(a),[int(1)]*len(a)],dtype=int).T
    epochs = mne.Epochs(raw,events,tmin=0,tmax=epoch_length,baseline=None,preload=True,proj=False)
    epochs.resample(500)
    
    annotation = pd.read_csv(annotation_file)
    spindles = annotation[annotation['Annotation'].apply(spindle_check)]
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
            
            connect_component_ratio = None
            number_connect_components = None
            average_component_size = None
            isolated_point = None
            end_point = None
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
def discritized_onset_label_manual(epochs,df,spindle_segment,front=300,back=100):
    start_times = epochs.events[:,0]
    end_times = epochs.events[:,0] + epochs.events[:,1]
    discritized_time_intervals = np.vstack((start_times,end_times)).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    temp=[]
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            temp.append([time_interval,spindle])
            if spindle_comparison(time_interval,spindle,spindle_segment):
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