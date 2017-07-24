# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:31:50 2017

@author: ning

"""

#import pandas as pd
import os
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from collections import Counter
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek,SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
try:
    function_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
except:
    function_dir = 'C:\\Users\\ning\\OneDrive\\python works\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
#import eegPipelineFunctions
try:
    file_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
#    file_dir = 'D:\\NING - spindle\\training set\\road_trip_29_channels\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
#    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip_29_channels\\'
    os.chdir(file_dir)
signal_features_dict=pickle.load(open(file_dir+'signal_features_data.p','rb'))
graph_features_dict= pickle.load(open(file_dir+'graph_features_data.p','rb'))
keys=['epoch_length ' + str(a) for a in np.arange(1.,5.,0.2)]
for ii,(key, dfs) in enumerate(zip(keys,signal_features_dict.values())):
    data = dfs.values   
    X, Y = data[:,:-1], data[:,-1]
    cv = KFold(n_splits=3,shuffle=True,random_state=12334)
    for jj,(train, test) in enumerate(cv.split(X,Y)):
        print('%s, cv %d'%(key,jj+1))
        ratio =  list(Counter(Y[train]).values())[1]/(list(Counter(Y[train]).values())[0]+list(Counter(Y[train]).values())[1])
#        clf = make_pipeline(SMOTETomek(random_state=12345,kind_smote='borderline2'),
#                            StandardScaler(),
#                            RandomForestClassifier(n_estimators=50,random_state=12345,criterion='gini',
#                            class_weight={1:1/(1-ratio)}))
#        clf = make_pipeline(SMOTETomek(random_state=12345,smote=SMOTE(k_neighbors=2,kind='borderline2')),
#                            StandardScaler(),
#                            XGBClassifier())
#        clf = make_pipeline(RandomUnderSampler(),RandomOverSampler(),
#                            StandardScaler(),
#                            XGBClassifier())
        clf=Pipeline([#('scaler',StandardScaler()),
                          ('estimator',XGBClassifier())])
        clf = make_pipeline(RandomUnderSampler(),SMOTE(k_neighbors=2,kind='borderline2'),
                            clf)
        clf = make_pipeline(RandomUnderSampler(),SMOTE(k_neighbors=2,kind='borderline2'),
#                            StandardScaler(),
                            RandomForestClassifier(n_estimators=50,random_state=12345,criterion='gini',
                            class_weight={1:0.99}))#1000/(1-ratio)
#        sample_weight= np.array([5 if i == 1 else 1 for i in Y])

        clf.fit(X[train],Y[train])
        true = Y[test];predic_prob=clf.predict_proba(X[test])[:,-1]
#        predict = predic_prob > ratio
        predict = clf.predict(X[test])
        fpr,tpr,T, = metrics.roc_curve(true,predic_prob)
        auc_score = metrics.auc(fpr,tpr)
        print(auc_score,'\n\n',
              metrics.classification_report(true, predict),'\n\n',
              metrics.matthews_corrcoef(true,predict),'\n\n',
              metrics.confusion_matrix(true,predict)/ metrics.confusion_matrix(true,predict).sum(axis=1)[:,np.newaxis],'\n\n') 
        
#######################################################################################################################################        
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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
directory_1 = os.listdir(file_dir)[0]
sub_dir = file_dir + directory_1 + '\\'
epoch_length = directory_1.split(' ')[1]
os.chdir(sub_dir)
sub_fold = os.listdir(sub_dir)[0]
sub_fold_dir = sub_dir + sub_fold + '\\'
os.chdir(sub_fold_dir)
sub = sub_fold[:-4]
day = sub_fold[4:][-4:]
cc_features, pli_features, plv_features, signal_features = [pd.read_csv(f) for f in os.listdir(sub_fold_dir) if ('csv' in f)]
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
df_graph = pd.concat([cc_features,pli_features,plv_features],axis=1)
df_graph['label']=label
df_combine = pd.concat([cc_features, pli_features, plv_features, signal_features],axis=1)
data = signal_features.values   
X, Y = data[:,:-1], data[:,-1]
cv = KFold(n_splits=10,shuffle=True,random_state=12334)
#from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
for jj,(train, test) in enumerate(cv.split(X,Y)):
    print('cv %d'%(jj+1))
    ratio =  list(Counter(Y[train]).values())[1]/(list(Counter(Y[train]).values())[0]+list(Counter(Y[train]).values())[1])
    clf = make_pipeline(SMOTETomek(random_state=12345,kind_smote='borderline2'),
                        StandardScaler(),
#                        LogisticRegressionCV(Cs=np.logspace(-3,3,7),scoring='roc_auc',max_iter=int(1e4),))
#                                             class_weight={1:1/(1-ratio)}))
                        RandomForestClassifier(n_estimators=40,random_state=12345,criterion='gini',#))
                        class_weight={1:1/(1-ratio)}))
    clf.fit(X[train],Y[train])
    true = Y[test];predic_prob=clf.predict_proba(X[test])[:,-1]
#    predict = predic_prob > ratio
    predict = clf.predict(X[test])
    fpr,tpr,T, = metrics.roc_curve(true,predic_prob)
    predict = predic_prob > T.mean()
    auc_score = metrics.auc(fpr,tpr)
    print(auc_score,'\n\n',
          metrics.classification_report(true, predict),'\n\n',
          metrics.matthews_corrcoef(true,predict),'\n\n',
          metrics.confusion_matrix(true,predict),'\n\n\n\n')
from sklearn.metrics import make_scorer
mcc = make_scorer(metrics.matthews_corrcoef)
cross_val_score(clf,X,Y,scoring=mcc,cv=cv)
from sklearn.model_selection import GridSearchCV
params = dict(smotetomek__kind_smote=['regular', 'borderline1', 'borderline2', 'svm'],
          randomforestclassifier__n_estimators=np.arange(20,50,5),
          randomforestclassifier__criterion=['gini','entropy'])
g=GridSearchCV(clf,param_grid=params,scoring=mcc,cv=cv)
g.fit(X,Y)













































        