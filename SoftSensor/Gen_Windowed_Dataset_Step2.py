#%%
from SubClass.ProcSoftSensorData import *
import pandas as pd
import numpy as np


#data, SubjNames, _ = LoadSegData([0, 2])
data, SubjNames, _ = LoadSegData(-1)
#SubjName = SubjNames[0]
#df_Subj = data[SubjName]
#MotRepNames = [*df_Subj.keys()] #컨테이너 타입의 데이터를 Unpacking 할 때 (*)


#%% Windowing
WindowSize = 200 #ms
#OverlapSize = 50 #ms
OverlapSize = 0 #ms
Tag = 'Sta' # ('Sta','Dyn',All')

####################################################################################
def CutWindowing(Sig, WindowSize, OverlapSize):
    # Size Validation Check
    if WindowSize < 1:
        WindowSize = 1
    if OverlapSize >= WindowSize:
        OverlapSize = WindowSize - 1
    IncSize = WindowSize - OverlapSize

    # Row of Sig is # of samples (Sig: n x ch)
    # Col of Sig is 3 of channels
    nLen = np.size(StableRegion, 0)
    #nCh = np.size(StableRegion, 1)

    fs = pd.DataFrame(columns=Sig.columns)    
    for win_s in np.arange(0,nLen,IncSize):
        if len(Sig[win_s:]) < WindowSize:
            break
        win_data = Sig.iloc[win_s:win_s+WindowSize,:]
        fs = fs.append(np.mean(win_data), ignore_index=True)



    return fs

####################################################################################
def SaveWindowedData(SubjName, Wdw_Data, WindowSize, OverlapSize, Tag):
    
    import pickle
    import bz2
    import gzip
    
    # Save Variable
    # with bz2.BZ2File('data/{}.pbz2'.format(subj), 'w') as f:
    #     pickle.dump(data, f)
    with gzip.GzipFile('data/Window/{}-{}.{}-{}.wdw'.format(SubjName, WindowSize, OverlapSize, Tag), 'w') as f:
        pickle.dump([SubjName, Wdw_Data, WindowSize, OverlapSize, Tag] , f)
   
    print("Windowed Dataset Saved.({})".format(SubjName))



# %% Main
# Loop for Cut Signals by windowing and return FeatureSet (fs)
windowed_data = {}
for SubjName in SubjNames: # Loop for Subject
    #Init fs & label
    try: 
        del(fs)
        del(label)
    except:
        pass

    MoTypes = data[SubjName].keys()    
    for MoType in MoTypes: # Loop for Motion Type (Static1-6, Dynamic1-6)
        #Currently, Dynamic MotionType is not used for the analysis
        if not((MoType.lower().find("static") == 0 and Tag.lower() == 'sta') or \
            (MoType.lower().find("dynamic") == 0 and Tag.lower() == 'dyn') or \
            (Tag.lower() == 'all')):
            continue
                
        IncRegion = "-S-"
        #IncRegion = "SSE" #SSE : Start Transition & Stable & End Transition
        #data[SubjName][MoType] : list which includes 12 motions (0:11, Rest:Num9)
        for MID, MName, StaTrans, StableRegion, EndTrans in data[SubjName][MoType]: # Visit each motion (tuple)
            if (IncRegion == "-S-"):
                SenData = StableRegion
            else:
                SenData = pd.concat([StaTrans, StableRegion, EndTrans], ignore_index=True)

            fs_ = CutWindowing(SenData, WindowSize, OverlapSize)            
            label_ = pd.DataFrame(index=fs_.index, columns=['Label', 'LabelName','MotionType'])
            label_['Label'] = MID
            label_['LabelName'] = MName
            label_['MotionType'] = MoType

            try:
                label = label.append(label_, ignore_index=True)
                fs = fs.append(fs_, ignore_index=True)
            except NameError: # if label and fs is not declared
                label = label_
                fs = fs_
    windowed_data = {'SID':SubjName, 'Label':label, 'FS':fs}
    SaveWindowedData(SubjName, windowed_data, WindowSize, OverlapSize, Tag+'_'+IncRegion)






# %%
#print(windowed_data)