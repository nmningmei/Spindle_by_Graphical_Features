# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:59:09 2017

@author: ning
"""

import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
filename = 'D:\\NING - spindle\\training set\\suj11_l2nap_day2.fif'
annotation_file = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\annotations\\suj11_nap_day2_edited_annotations.txt'
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions

epochs,label,temp = eegPipelineFunctions.get_data_ready(filename,channelList,annotation_file,)

epochFeature = eegPipelineFunctions.featureExtraction(epochs,)
epochFeature = pd.DataFrame(epochFeature)
connectivity = eegPipelineFunctions.connectivity(epochs)
t = 0.8
connectivity = np.array(connectivity)
cc = connectivity[:,-1,:,:]

adj = eegPipelineFunctions.thresholding(t,cc)
graphFeature = eegPipelineFunctions.extractGraphFeatures(adj)
before_feature = eegPipelineFunctions.extractGraphFeatures(cc)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,matthews_corrcoef,make_scorer
lcv = np.logspace(-3,3,7)
cv = KFold(n_splits=10,shuffle=True,random_state=123)
clf = Pipeline([('scaler',StandardScaler()),
               ('estimator',LogisticRegressionCV(Cs=lcv,cv=cv,max_iter=1e4))])
#results = []
#for train,test in cv.split(graphFeature.values):
#    x_train = graphFeature.values[train]
#    y_train = label[train]
#    clf.fit(x_train,y_train)
#    x_test = graphFeature.values[test]
#    y_test = label[test]
#    
#    results.append(f1_score(y_test,clf.predict(x_test)))
features = pd.concat([graphFeature,before_feature],axis=1)
results= cross_val_score(clf,features.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
results = cross_val_score(clf,graphFeature.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
results = cross_val_score(clf,before_feature.values,label,scoring='roc_auc',cv=cv)
print(np.mean(results))
#######################################################################################################################################
import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
filename = 'D:\\NING - spindle\\training set\\suj11_l2nap_day2.fif'
annotation_file = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\annotations\\suj11_nap_day2_edited_annotations.txt'
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions
raw_dir = 'D:\\NING - spindle\\training set\\'
# get EEG files that have corresponding annotations
raw_files = []
for file in [f for f in os.listdir(raw_dir) if ('txt' in f)]:
    sub = int(file.split('_')[0][3:])
    if sub < 11:
        day = file.split('_')[1][1]
        day_for_load = file.split('_')[1][:2]
    else:
        day = file.split('_')[2][-1]
        day_for_load = file.split('_')[2]
    raw_file = [f for f in os.listdir(raw_dir) if (file.split('_')[0] in f) and (day_for_load in f) and ('fif' in f)]
    if len(raw_file) != 0:
        raw_files.append([raw_dir + raw_file[0],raw_dir + file])
        
raw_file, annotation_file = raw_files[0]
l_freq = 11;h_freq = 16
epoch_length = 2; overlapping = 0.2
raw = mne.io.read_raw_fif(raw_file,preload=True)
if channelList is not None:
    raw.pick_channels(channelList)
else:
    raw.drop_channels(['LOc','ROc'])
raw.filter(l_freq,h_freq,filter_length='10s', l_trans_bandwidth=0.1, h_trans_bandwidth=0.5,n_jobs=4,)
a=epoch_length - overlapping * 2
events = mne.make_fixed_length_events(raw,id=1,duration=a)
epochs = mne.Epochs(raw,events,tmin=0,tmax=epoch_length,baseline=None,preload=True,proj=False)
epochs.resample(64)
annotation = pd.read_csv(annotation_file)
spindles = annotation[annotation['Annotation'].apply(eegPipelineFunctions.spindle_check)]

time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
peak_time={} #preallocate
sfreq=raw.info['sfreq']
mph,mpl = {},{}
moving_window_size = sfreq
lower_threshold = 0.4;higher_threshold=3.5
front,back=300,100
l_bound,h_bound = 0.5,2
tol=1;syn_channels=3


from scipy import signal,stats

def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = signal.gaussian(window_size,(window_size/.68)/2)
  return np.sqrt(np.convolve(a2, window, 'same')/len(a2)) * 1e2
def trimmed_std(a,p):
    temp = stats.trimboth(a,p/2)
    return np.std(temp)
for ii, names in enumerate(channelList):

    peak_time[names]=[]
    segment,_ = raw[ii,:]
    RMS[ii,:] = window_rms(segment[0,:],moving_window_size) 
    mph[names] = stats.trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) 
    mpl[names] = stats.trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05)
    pass_ = RMS[ii,:] > mph[names]#should be greater than then mean not the threshold to compute duration

    up = np.where(np.diff(pass_.astype(int))>0)
    down = np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    if down[0] < up[0]:
        down = down[1:]
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    for pairs in C.T:
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
            if np.max(SegmentForPeakSearching) < mpl[names]:
                temp_temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time[names].append(temp_temp_time[ints_temp])

peak_time['mean']=[];peak_at=[];duration=[]
RMS_mean=stats.hmean(RMS)
mph['mean'] = stats.trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS_mean,0.05)
mpl['mean'] = stats.trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS_mean,0.05)
pass_ =RMS_mean > mph['mean']
up = np.where(np.diff(pass_.astype(int))>0)
down= np.where(np.diff(pass_.astype(int))<0)
up = up[0]
down = down[0]
if down[0] < up[0]:
    down = down[1:]
if (up.shape > down.shape) or (up.shape < down.shape):
    size = np.min([up.shape,down.shape])
    up = up[:size]
    down = down[:size]
C = np.vstack((up,down))
for pairs in C.T:
    if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
        SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
        if np.max(SegmentForPeakSearching)< mpl['mean']:
            temp_time = time[pairs[0]:pairs[1]]
            ints_temp = np.argmax(SegmentForPeakSearching)
            peak_time['mean'].append(temp_time[ints_temp])
            peak_at.append(SegmentForPeakSearching[ints_temp])
            duration_temp = time[pairs[1]] - time[pairs[0]]
            duration.append(duration_temp)
time_find=[];mean_peak_power=[];Duration=[];
for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
    temp_timePoint=[]
    for ii, names in enumerate(channelList):
        try:
            temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
        except:
            temp_timePoint.append(item + 2)
    try:
        if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
            time_find.append(float(item))
            mean_peak_power.append(PEAK)
            Duration.append(duration_time)
    except:
        pass
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
# sleep stage
temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
# seperate out stage 2
stages = annotation[annotation.Annotation.apply(stage_check)]
On = stages[::2];Off = stages[1::2]
stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
if abs(np.diff(stage_on_off[0]) - 30) < 2:
    pass
else:
    On = stages[1::2];Off = stages[::2]
    stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
    for on_time,off_time in stage_on_off:
        if intervalCheck([on_time,off_time],single_time_find,tol=tol):
            temp_time_find.append(single_time_find)
            temp_mean_peak_power.append(single_mean_peak_power)
            temp_duration.append(single_duration)
time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
result = pd.DataFrame({'Onset':time_find,'Duration':Duration,'Annotation':['spindle']*len(Duration)})
auto_label,_ = eegPipelineFunctions.discritized_onset_label_auto(epochs,raw,result,epoch_length=epoch_length,)
manual_label,_ = eegPipelineFunctions.discritized_onset_label_manual(epochs,raw,epoch_length=epoch_length, df=spindles,spindle_duration=2)
full_prop=[] 
data = epochs.get_data()       
for d in data:    
    temp_p=[]
    #fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
    for ii,(name) in enumerate(zip(channelList)):#,ax.flatten())):
        rms = window_rms(d[ii,:],500)
        l = stats.trim_mean(rms,0.05) + lower_threshold * trimmed_std(rms,0.05)
        h = stats.trim_mean(rms,0.05) + higher_threshold * trimmed_std(rms,0.05)
        prop = (sum(rms>l)+sum(rms<h))/(sum(rms<h) - sum(rms<l))
        if np.isinf(prop):
            prop = (sum(rms>l)+sum(rms<h))
        temp_p.append(prop)
    full_prop.append(temp_p)
full_prop = np.array(full_prop)
psds,freq = mne.time_frequency.psd_multitaper(epochs,fmin=l_freq,fmax=h_freq,tmin=0,tmax=epoch_length,low_bias=True,)
psds = 10* np.log10(psds)
features = pd.DataFrame(np.concatenate((full_prop,psds.max(2),freq[np.argmax(psds,2)]),1))
signalFeature = eegPipelineFunctions.featureExtraction(epochs,)
signalFeature = pd.DataFrame(signalFeature)
signalFeature['label']=auto_label
signalFeature_m = signalFeature.copy()
signalFeature_m['label']=manual_label
from sklearn.model_selection import cross_val_score
clf = eegPipelineFunctions.make_pipeline(RandomUnderSampler(),RandomOverSampler(),
                            StandardScaler(),
                            eegPipelineFunctions.XGBClassifier())
#                            LogisticRegressionCV(Cs=np.logspace(-4,6,11),cv=5,tol=1e-7,max_iter=int(1e7)))
X,Y = signalFeature.values[:,:-1],signalFeature.values[:,-1]
xx = np.concatenate((X,features),axis=1)
#auto_proba = clf.predict_proba(signalFeature.values[:,:-1])[:,-1]
aa = cross_val_score(clf,xx,Y,scoring='roc_auc',cv=10)

X,Y = signalFeature_m.values[:,:-1],signalFeature_m.values[:,-1]
mm = cross_val_score(clf,features,Y,scoring='roc_auc',cv=10)
print(aa.mean(),mm.mean())



















