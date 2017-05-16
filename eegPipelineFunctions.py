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
from scipy.sparse.csgraph import connected_components
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


def get_data_ready(filename,channelList,l_freq=11,h_freq=16,epoch_length=5,overalapping=0.2,
                   ):
    raw = mne.io.read_raw_fif(filename,preload=True)
    raw.pick_channels(channelList)
    raw.filter(l_freq,h_freq)
    a = np.arange(0,raw.times[-1],epoch_length - epoch_length*overalapping)
    events = np.array([a,[epoch_length]*len(a),[int(1)]*len(a)],dtype=int).T
    epochs = mne.Epochs(raw,events,tmin=0,tmax=epoch_length,baseline=None,preload=True,proj=False)
    epochs.resample(500)
    return epochs

def featureExtraction(epochs):
    features = ['mean','variance','delta_mean',
          'delta_variance','change_variance',
         'activity','mobility','complexity','skewness_of_amplitude_spectrum',
         'spectral_entropy']
    epochFeatures = {name:[] for name in features}
    for ii, epoch_data in enumerate(epochs):
        epoch_data  = epoch_data[:,:-1].T
        print('computing features for epoch %d' %(ii+1))
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
        print('computing connectivity for epoch %d' %(ii+1))
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
def extractGraphFeatures(adjacency):
    for ii,a in enumerate(adjacency):
        G = nx.from_numpy_matrix(a)
        average_degree = nx.average_neighbor_degree(G)
        average_degree = np.mean([v for v in average_degree.values()])
        
        Clustering_coefficient = nx.average_clustering(G)
        
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
        
        
