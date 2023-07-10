#%% Load Data
from SubClass.ProcSoftSensorData import *
import numpy as np
import pandas as pd
from StaticMotionMapInfo_200901_bySST_Final import MotionNames, StaticMotionMap, CheckDataIntegrity_StaticMotionMap


# %%
def SaveSegMotionData(SubjName, Seg_Data):
    
    import pickle
    import bz2
    import gzip
    
    # Save Variable
    # with bz2.BZ2File('data/{}.pbz2'.format(subj), 'w') as f:
    #     pickle.dump(data, f)
    with gzip.GzipFile('data/Seg/{}.seg'.format(SubjName), 'w') as f:
        pickle.dump(Seg_Data, f)
   
    print("Segmented Data Saved.({})".format(SubjName))
    
    
    
    
# %%
def Cut_StaticMotion(SubjName, StaticNames):  
    raw_data = data[SubjName]    
    
    # Init Dict
    SegData = {SubjName: dict()}
    
    for MotRepName in StaticNames:
        df_Mot = raw_data[MotRepName]
        Sig_Col_Names = df_Mot.columns[1:11] # A0 - A9
        #df_Mot = df_Subj[StaticNames[0]]
        
        # Init MotRepName ('Static 1', ...)
        SegData[SubjName][MotRepName] = list()
     
        # Static Motion Map
        try:
            SampleIndex = StaticMotionMap[SubjName][MotRepName]
        except:
            SampleIndex = []     
    
        for tup_sample in SampleIndex:
            Index = tup_sample[0] # Motion Index; Please use MotionNames for the String Name
            x_Start_M = tup_sample[1] #Start of Motion (Transient)
            x_Start_S = tup_sample[2] #Start of Stable Motion
            x_End_S = tup_sample[3] #End of Stable Motion
            x_End_M = tup_sample[4] #End of Motion (Transient)
        
            Name = MotionNames[Index]
            Start_Transient = df_Mot[Sig_Col_Names][x_Start_M:x_Start_S]    
            Stable_Motion = df_Mot[Sig_Col_Names][x_Start_S:x_End_S]
            End_Transient = df_Mot[Sig_Col_Names][x_End_S:x_End_M]
    
            tmp = (Index, Name, Start_Transient, Stable_Motion, End_Transient)
            
            # Update the Segmented Data
            SegData[SubjName][MotRepName].append(tmp)
        
    SaveSegMotionData(SubjName, SegData)


        

    
    
    
# %% Main
data, SubjNames, _ = LoadSoftSensorData(-1)

SubjName = SubjNames[0]
#df_Subj = data[SubjName]
#TestNames = [*df_Subj.keys()]
is_Static = lambda x: x.lower().startswith('static')
#StaticNames = list(filter(is_Static, TestNames))



# Static Motion Map - Integrity Check
CheckDataIntegrity_StaticMotionMap(StaticMotionMap)




for SubjName in SubjNames:
    
    TestNames = [*data[SubjName].keys()]
    StaticNames = list(filter(is_Static, TestNames))
    Cut_StaticMotion(SubjName, StaticNames)
   