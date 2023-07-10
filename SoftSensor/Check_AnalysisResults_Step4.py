import SubClass.ProcSoftSensorData as sub
import SubClass.ML_Analysis as ML
import SubClass.PlotLib as PlotLib
import SubClass.UserLib as UserLib

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import time
from datetime import timedelta 

import sys
import itertools

from StaticMotionMapInfo_200901_bySST_Final import MotionNames_v2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Defaults
FolderName = 'Results'
PlotDirName = 'Plots'
AnalTag = "sta_-s-_allsubj_210212"


# %% Check Arguments
import sys
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
if len(sys.argv) == 2:
    AnalTag = sys.argv[1]

print('Analysis Tag:', str(AnalTag))


# %% Main of Check Analysis Restuls
(results, perfs, ClassLabelNames, pipe_mdls) = ML.Load_ML_Analysis_Results(AnalTag)
print('ClassLabelNames:', str(ClassLabelNames))


SaveDirName = PlotDirName + '/' + AnalTag
UserLib.MakeFolder(SaveDirName)


tag = AnalTag
tag = tag.replace("/", "_") # Remove Sys Character
tag = tag.replace("\\", "_") # Remove Sys Character
fn = SaveDirName + '/output_%s.txt' % (tag)

dic_y_true = {}
dic_y_pred = {}

original_stdout = sys.stdout
with open(fn, 'w') as f:
    sys.stdout = f

    ML.ShowPerformnaceTable(perfs, pipe_mdls)


    for clf_name, ret in results.items():
        print('-----------------------------------------')
        print(str(clf_name))
        print('-----------------------------------------')        
        
        tmp = list(zip(*ret))
        y_true_all = list(itertools.chain(*tmp[0]))
        y_pred_all = list(itertools.chain(*tmp[1]))
        ML.ShowResult(y_true_all, y_pred_all)

        dic_y_true[clf_name] = y_true_all
        dic_y_pred[clf_name] = y_pred_all
        
        # Draw Confusion Matrix 
        plt.close('all')
        PlotLib.Plot_ConfMat_EachCfy(y_true_all, y_pred_all, ClassLabelNames, f"%s\n\n\nConfusion Matrix - %s" % (tag, clf_name))            
        plt.savefig(SaveDirName + f"\ConfMat-%s.png"%(clf_name), dpi=600)


    # Draw Precision/Recall/F1
    PlotLib.Plot_AccBox_AllCfy(perfs['ACC'], (0.75,1), f"%s" % tag)
    plt.savefig(SaveDirName + "\ACC-Box.png", dpi=600)
    PlotLib.Plot_Bars_AllCLF_AllLBL(dic_y_true, dic_y_pred, ClassLabelNames, precision_score, "Precision", f"%s" % tag)
    plt.savefig(SaveDirName + "\Precision.png", dpi=600)
    PlotLib.Plot_Bars_AllCLF_AllLBL(dic_y_true, dic_y_pred, ClassLabelNames, recall_score, "Recall", f"%s" % tag)
    plt.savefig(SaveDirName + "\Recall.png", dpi=600)
    PlotLib.Plot_Bars_AllCLF_AllLBL(dic_y_true, dic_y_pred, ClassLabelNames, f1_score, "F1 Score", f"%s" % tag)
    plt.savefig(SaveDirName + "\F1Score.png", dpi=600)


sys.stdout = original_stdout
