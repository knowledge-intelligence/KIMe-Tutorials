# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:32:07 2020

@author: sstmir

@comments: "bySubj" means that K-fold is performed on subject's level, not all samples.

"""
#%% Start
import SubClass.ProcSoftSensorData as sub
import SubClass.ML_Analysis as ML
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import time
from datetime import timedelta


ResultDirName = 'Results'
PlotDirName = 'Plots'
AnalysisTag = "sta_-s-_allsubj_210212"

# Set Parameters
ML.n_GridSearchCV = 3 # Set CV value as 2 for GridSearchCV

## PCA or None
ml_pipe_generator = ML.Gen_ML_Pipe  # usin all features
# ml_pipe_generator = ML.Gen_ML_PCA_Pipe
#ML.b_DR_PCA_Fixed = False
#ML.n_DR_PCA_FixedPCs = 6


## Load Data
#data, SubjNames, _ = sub.LoadWdwData(200,0,'sta',[0,2])
data, SubjNames, _ = sub.LoadWdwData(200,0,'sta_-s-', SubjectNo=-1) # Load All Subjects (15)

## Prepare Data
from StaticMotionMapInfo_200901_bySST_Final import MotionNames, MotionNames_v2
ClassLabelNames = MotionNames_v2

## data of all subjects for Model Development by GridSearchCV
SID = list(data.keys())

## X & y
data_y = pd.concat([data[id]['Label'] for id in SID], ignore_index=True)
data_y = data_y['Label']
data_X = pd.concat([data[id]['FS'] for id in SID], ignore_index=True)

## Handling Imbalanced Data
data_X_re, data_y_re, sampler = ML.De_Imbalance_byOver(data_X, data_y)




## Find The Best Classifiers (Model Development)
mdl = {}
mdl['KNN'] = ML.Get_KNN(ml_pipe_generator, data_X_re, data_y_re)
mdl['NB'] = ML.Get_NB(ml_pipe_generator, data_X_re, data_y_re)
mdl['LDA'] = ML.Get_LDA(ml_pipe_generator, data_X_re, data_y_re)
mdl['QDA'] = ML.Get_QDA(ml_pipe_generator, data_X_re, data_y_re)
mdl['SVM'] = ML.Get_SVM(ml_pipe_generator, data_X_re, data_y_re)
mdl['RF'] = ML.Get_RF(ml_pipe_generator, data_X_re, data_y_re)




## K(5)-fold CV
from sklearn.model_selection import StratifiedKFold
#CV = StratifiedKFold(n_splits=5, shuffle=False)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



## X & y by Subject
X_subj, y_subj = {}, {}
for key, subj in data.items():
    X_subj[key] = subj['FS']
    y_subj[key] = subj['Label']['Label']

## Run Classification & Calculate Performance
## Need to CV @ Subject Level (Not @ Sample Level for higher performance)
clf_results, tr_te_indexs = ML.Run_PerformanceBySubject(mdl, X_subj, y_subj, CV)
clf_perfs = ML.Cal_Perfermance(clf_results)
ML.ShowPerformnaceTable(clf_perfs, mdl)

## Save Results
ML.Save_ML_Analysis_Results(data_X_re, data_y_re, ClassLabelNames, tr_te_indexs, sampler, CV,
                mdl, clf_results, clf_perfs, AnalysisTag)
