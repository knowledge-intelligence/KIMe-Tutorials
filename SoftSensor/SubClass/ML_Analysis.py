import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import time
from datetime import timedelta
import shelve
import os

import pathlib # Python 3.5+
#pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True) 


# %%
#########################################################################################
## Declare Parameters
#########################################################################################
n_GridSearchCV = 2      # Default CV value of Grid Search CV
n_DR_PCA_PCs = 6           # Default PCA # of Components
n_KNN_K = 10            # Default Max K of KNN
n_LDA_n = 10            # Default Max n_components of LDA

b_DR_PCA_Fixed = False
n_DR_PCA_FixedPCs = 6


# %%
#########################################################################################
## Save, Load, Show
#########################################################################################
def Save_ML_Analysis_Results(train, test, ClassLabelNames, tr_te_indexs, sampler, CV,
                pipe_mdls, clf_results, clf_perfs, AnalTag, ResultDirName = 'Results'):

    from datetime import datetime

    pathlib.Path(ResultDirName).mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    fn = ResultDirName + "/ML_Results_" + AnalTag
        
    with shelve.open(fn) as results:        
        results['Gen_DateTime'] = now
        results['Gen_DateTimeStr'] = now.strftime("%y%m%d-%H%M%S")
        results['TrainSet'] = train
        results['TestSet'] = test
        results['ClassLabelNames'] = ClassLabelNames
        results['Tr_Te_Indexs'] = tr_te_indexs
        results['Sampler'] = sampler #Sampler (e.g. RandOverSampler)
        results['CV'] = CV
        results['Pipe_Models'] = pipe_mdls
        results['Results'] = clf_results #ret_KNN, etc.
        results['Performances'] = clf_perfs #acc, etc.

        results.close()


def Load_ML_Analysis_Results(AnalTag, ResultDirName = 'Results'):
    fn = ResultDirName + "/ML_Results_" + AnalTag

    with shelve.open(fn, flag='r') as results:
        ClassLabelNames = results['ClassLabelNames']
        pipe_mdls = results['Pipe_Models']
        clf_results = results['Results'] #ret_KNN, etc.
        clf_perfs = results['Performances'] #acc, etc.
     
    return (clf_results, clf_perfs, ClassLabelNames, pipe_mdls)    


def ShowResult(y_true, y_pred):
    precision, recall, f1score, support = \
        precision_recall_fscore_support(y_true, y_pred)

    print('Precision (per Label): {}'.format(precision))
    print('Recall (per Label): {}'.format(recall))
    print('F1 Score (per Label): {}'.format(f1score))
    # The number of occurrences of each label in y_true
    print('Support (# per True Label): {}'.format(support))
    print()
    print()    
    print('Accuracy (Averaged): ' + str(accuracy_score(y_true, y_pred)))
    print('Precision (Averaged): ' + str(precision_score(y_true, y_pred, average='weighted')))
    print('Recall (Averaged): ' + str(recall_score(y_true, y_pred, average='weighted')))
    print('F1 Score (Averaged): ' + str(f1_score(y_true, y_pred, average='weighted')))
    print()
    print()
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    print()
    print()
    print('Classification Report')
    print(classification_report(y_true, y_pred, digits=3))


def ShowPerformnaceTable(Pref, mdl):
    
    print('Model Parameters: ')
    for key, clf in mdl.items():
        print(key, end=': ')
        print(clf.get_params())
    print()
    print()

    print('Accuracy (per CV): ')
    for key, value in Pref['ACC'].items():
        print(key, end=', ')
        print(*value, sep=', ')
    print()
    print()
    
    print('Precision (per CV): ')
    for key, value in Pref['PRE'].items():
        print(key, end=', ')
        print(*value, sep=', ')
    print()
    print()

    print('Recall (per CV): ')
    for key, value in Pref['REC'].items():
        print(key, end=', ')
        print(*value, sep=', ')
    print()
    print()

    print('F1 Score (per CV): ')
    for key, value in Pref['F1'].items():
        print(key, end=', ')
        print(*value, sep=', ')
    print()
    print()


# %%
#########################################################################################
## Declare Basic Functions
#########################################################################################

# def Get_Xy(DataSet):
#     ## For Multi-(or Binary) Lables
#     X = DataSet.drop('Label', axis=1)
#     y = DataSet['Label'].copy()
#     return (X, y)
    
def De_Imbalance_byOver(X, y):
    ## Naive random over-sampling (for the less number of the control group)
    ## This was performed only on the train set
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return (X_resampled, y_resampled, ros)

def Gen_ML_Pipe(clf):    
    ## StandardScaler (Normalization - ZScore)
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', clf)])
    return pipe

def Gen_ML_PCA_Pipe(clf):    
    ## StandardScaler (Normalization - ZScore)
    ## PCA for Dimensionality Reduction
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('dr', PCA()), ('clf', clf)])
    return pipe


def Get_Best_Mdl(X, y, pipe_mdl, param_grid):

    if len(param_grid) == 0: # No need to model parameter optimization (e.g. NB)
        print('No Parameters To Opimize.')
        return pipe_mdl.set_params(param_grid)


    # Model Development by GridSearchCV
    grid_search = GridSearchCV(pipe_mdl, param_grid, cv=n_GridSearchCV, n_jobs=4,
                            scoring='accuracy', return_train_score=False, 
                            refit=False, verbose = 4)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    return grid_search

def Run_Performance(pipe_mld_dict, X, y, CV):
    ## Leave-One-Out by Subject (it takes too many run-time)
    # from sklearn.model_selection import LeaveOneOut
    # CV = LeaveOneOut()
    # from sklearn.model_selection import StratifiedKFold
    # CV = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    # CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Start Time
    start_time = time.time()    
    
    cvindexs = []
    results = dict((k, []) for k in pipe_mld_dict.keys()) # init dic w/ different empty list    

    CV_Loop_n = 0
    for train_index, test_index in CV.split(X, y): # CV Split Dataset
        CV_Loop_n = CV_Loop_n + 1

        cvindexs.append((train_index, test_index))
        
        X_train = X.loc[train_index, :]
        y_train = y.loc[train_index] # Panda Series
        X_test = X.loc[test_index, :]
        y_true = y.loc[test_index] # y_test

        for name, pipe_mdl in pipe_mld_dict.items():
            start_loop_time = time.time()

            pipe_mdl.fit(X_train, y_train)
            y_pred = pipe_mdl.predict(X_test)        
            results[name].append((y_true.to_numpy(), y_pred)) # Series -> ndarray
            ShowResult(y_true.to_numpy(), y_pred)
    
            elapsed_time_secs = time.time() - start_loop_time    
            msg = "[Interim %d]Execution took: %s secs" % (CV_Loop_n, timedelta(seconds=round(elapsed_time_secs)))
            print(msg)

    #End Time
    elapsed_time_secs = time.time() - start_time
    msg = "[Final-Total]Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs))    

    return results, cvindexs 


def Run_PerformanceBySubject(pipe_mld_dict, X, y, CV):
    #Start Time
    start_time = time.time()    
    
    cvindexs = []
    results = dict((k, []) for k in pipe_mld_dict.keys()) # init dic w/ different empty list    

    SID = list(X.keys())
    y_placeholder = np.zeros(len(y))
    CV_Loop_n = 0
    for train_index, test_index in CV.split(X, y_placeholder): # CV Split Dataset
        CV_Loop_n = CV_Loop_n + 1

        cvindexs.append((train_index, test_index))
        
        y_train = pd.concat([y[SID[i]] for i in train_index], ignore_index=True)
        X_train = pd.concat([X[SID[i]] for i in train_index], ignore_index=True)
        y_true = pd.concat([y[SID[i]] for i in test_index], ignore_index=True)
        X_test = pd.concat([X[SID[i]] for i in test_index], ignore_index=True)

        for name, pipe_mdl in pipe_mld_dict.items():
            start_loop_time = time.time()

            pipe_mdl.fit(X_train, y_train)
            y_pred = pipe_mdl.predict(X_test)        
            results[name].append((y_true.to_numpy(), y_pred)) # Series -> ndarray
            ShowResult(y_true.to_numpy(), y_pred)
    
            elapsed_time_secs = time.time() - start_loop_time    
            msg = "[Interim %d]Execution took: %s secs" % (CV_Loop_n, timedelta(seconds=round(elapsed_time_secs)))
            print(msg)

    #End Time
    elapsed_time_secs = time.time() - start_time
    msg = "[Final-Total]Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs))    

    return results, cvindexs


def Cal_Perfermance(results):
    acc, pre, rec, f1, sup = {}, {}, {}, {}, {}
    for name, result in results.items():
        acc[name], pre[name], rec[name], f1[name], sup[name] = \
            [], [], [], [], []

        for num, cv_val in enumerate(result, start=0):
            num
            y_true, y_pred = cv_val
            
            precision, recall, f1score, support = \
                precision_recall_fscore_support(y_true, y_pred, average='weighted')

            acc[name].append( accuracy_score(y_true, y_pred) )
            pre[name].append( precision )
            rec[name].append( recall )
            f1[name].append( f1score )
            sup[name].append( support )

    pref = {'ACC': acc, 'PRE': pre, 'REC': rec, 'F1': f1, 'Support': sup}
    return pref


# %%
#########################################################################################
## Declare DR PCA Handler
#########################################################################################
def Handle_DR_PCA_Para(pipe_mdl, param_grid):
    if b_DR_PCA_Fixed:
        return Handle_DR_PCA_Para_Fixed(pipe_mdl, param_grid)
    else:
        return Handle_DR_PCA_Para_Range(pipe_mdl, param_grid)

def Handle_DR_PCA_Para_Range(pipe_mdl, param_grid):
    print('DR_PCA_Para_Range(1-%d).' % n_DR_PCA_PCs)
    if 'dr__n_components' in pipe_mdl.get_params():
        for para in param_grid:
            para.update( {'dr__n_components': range(1,n_DR_PCA_PCs+1)} )
    return param_grid

def Handle_DR_PCA_Para_Fixed(pipe_mdl, param_grid):
    print('DR_PCA_Para_Fixed(%d).' % n_DR_PCA_FixedPCs)
    if 'dr__n_components' in pipe_mdl.get_params():
        for para in param_grid:
            para.update( {'dr__n_components': [n_DR_PCA_FixedPCs]} )
    return param_grid


#########################################################################################
## Declare Each Classifiers
#########################################################################################
def Get_KNN(pipe_generator, X, y):
    from sklearn.neighbors import KNeighborsClassifier

    print('Analyzing KNN...')
    
    pipe_mdl = pipe_generator(KNeighborsClassifier())

    # Determine KNN Model Parameter
    param_grid = [{}]
    if 'clf__n_neighbors' in pipe_mdl.get_params() and isinstance(pipe_mdl['clf'], KNeighborsClassifier):
        param_grid = [{'clf__n_neighbors': range(2,n_KNN_K)}]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)
    
    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)

    return pipe_mdl


def Get_NB(pipe_generator, X, y):
    from sklearn.naive_bayes import GaussianNB

    print('Analyzing NB...')
    
    pipe_mdl = pipe_generator(GaussianNB())

    # Determine NB Model Parameter
    param_grid = [{}]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)
    
    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    if grid_search == pipe_mdl: # which means no parameters to optimize
        return pipe_mdl    
    

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)
    return pipe_mdl



def Get_SVM(pipe_generator, X, y):
    from sklearn.svm import SVC

    print('Analyzing SVM...')
    
    pipe_mdl = pipe_generator(SVC())

    # Determine SVM Model Parameter
    param_grid = [{}]
    if isinstance(pipe_mdl['clf'], SVC):
        # Example of Parameter Range
        # Scikit learn: C_range = np.logspace(-2, 10, 13)
        # Scikit learn: gamma_range = np.logspace(-9, 3, 13)
        C_range = 10**np.linspace(-3,7,7+3+1) # np.logspace(-2, 5, 10)
        gamma_range = 10**np.linspace(-5,5,5+5+1) # np.logspace(-5, 3, 10)    
        param_grid = [ {'clf__kernel': ['rbf'], 'clf__gamma': gamma_range, 'clf__C': C_range}, \
                        {'clf__kernel': ['linear'], 'clf__C': C_range} ]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)    

    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)

    return pipe_mdl


def Get_LDA(pipe_generator, X, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    print('Analyzing LDA...')
    
    pipe_mdl = pipe_generator(LinearDiscriminantAnalysis())
    
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_LDA_n = np.min([n_classes - 1, n_features])

    # Determine LDA Model Parameter
    param_grid = [{}]
    if 'clf__n_components' in pipe_mdl.get_params() and isinstance(pipe_mdl['clf'], LinearDiscriminantAnalysis):
        param_grid = [{'clf__n_components': range(1, n_LDA_n+1)}]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)
    
    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)

    return pipe_mdl


def Get_QDA(pipe_generator, X, y):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    print('Analyzing QDA...')
    
    pipe_mdl = pipe_generator(QuadraticDiscriminantAnalysis())

    # Determine QDA Model Parameter
    param_grid = [{}]
    if 'clf__reg_param' in pipe_mdl.get_params() and isinstance(pipe_mdl['clf'], QuadraticDiscriminantAnalysis):
        param_grid = [{'clf__reg_param': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)
    
    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)

    return pipe_mdl

def Get_RF(pipe_generator, X, y):
    from sklearn.ensemble import RandomForestClassifier

    print('Analyzing RF...')
    
    pipe_mdl = pipe_generator(RandomForestClassifier())

    ## Random Search of Hyper Parameters
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1500, num = 5)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    max_features = ['auto']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth = [int(x) for x in np.linspace(10, 30, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    min_samples_split = [2]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    min_samples_leaf = [1]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    bootstrap = [True]

    # Determine RF Model Parameter    
    param_grid = [{}]
    if isinstance(pipe_mdl['clf'], RandomForestClassifier):
        param_grid = [{'clf__n_estimators': n_estimators, 
        'clf__max_features': max_features,
        'clf__max_depth': max_depth,
        'clf__min_samples_split': min_samples_split,
        'clf__min_samples_leaf': min_samples_leaf,
        'clf__bootstrap': bootstrap}]
    param_grid = Handle_DR_PCA_Para(pipe_mdl, param_grid)
    
    grid_search = Get_Best_Mdl(X, y, pipe_mdl, param_grid)

    # Set Model w/ Best Parameters
    # To convert 'dict' to 'kwargs', e.g., pipe_mdl.set_params(clf__n_neighbors = 2)
    # Use the double-star (aka double-splat) operator:
    pipe_mdl.set_params(**grid_search.best_params_)

    return pipe_mdl