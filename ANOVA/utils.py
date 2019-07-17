#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:30:19 2018

@author: nmei
"""
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     LeaveOneGroupOut,
                                     cross_val_score)
from sklearn.metrics import roc_auc_score,matthews_corrcoef
from collections import Counter
try:
    from tqdm import tqdm
except:
    print("why is tqdm not installed?")
try:
    import pymc3 as pm
except:
    print("you don't have pymc3 or you haven't set up the environment")
#import theano.tensor as t


def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 5000,one_tail = False):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    """
    import numpy as np
    experiment = np.mean(x) # the mean of the observations in the experiment
    experiment_diff = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    # newexperiment = np.mean(experiment_diff) # just look at the new mean and make sure it is at the baseline
    # simulate/bootstrap null hypothesis distribution
    # 1st-D := number of sample same as the experiment
    # 2nd-D := within one permutation resamping, we perform resampling same as the experimental samples,
    # but also repeat this one sampling n_permutation times
    # 3rd-D := repeat 2nd-D n_ps times to obtain a distribution of p values later
    temp = np.random.choice(experiment_diff,size=(x.shape[0],n_permutation,n_ps),replace=True)
    temp = temp.mean(0)# take the mean over the sames because we only care about the mean of the null distribution
    # along each row of the matrix (n_row = n_permutation), we count instances that are greater than the observed mean of the experiment
    # compute the proportion, and we get our p values
    
    if one_tail:
        ps = (np.sum(temp >= experiment,axis=0)+1.) / (n_permutation + 1.)
    else:
        ps = (np.sum(np.abs(temp) >= np.abs(experiment),axis=0)+1.) / (n_permutation + 1.)
    return ps
def resample_ttest_2sample(a,b,n_ps=100,n_permutation=5000,one_tail=False,match_sample_size = True,):
    # when the N is matched just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,baseline=0,n_ps=n_ps,n_permutation=n_permutation,one_tail=one_tail)
        return ps
    else: # when the N is not matched
        difference              = np.mean(a) - np.mean(b)
        concatenated            = np.concatenate([a,b])
        np.random.shuffle(concatenated)
        temp                    = np.zeros((n_permutation,n_ps))
        # the next part of the code is to estimate the "randomized situation" under the given data's distribution
        # by randomized the items in each group (a and b), we can compute the chance level differences
        # and then we estimate the probability of the chance level exceeds the true difference 
        # as to represent the "p value"
        try:
            iterator            = tqdm(range(n_ps),desc='ps')
        except:
            iterator            = range(n_ps)
        for n_p in iterator:
            for n_permu in range(n_permutation):
                idx_a           = np.random.choice(a    = [0,1],
                                                   size = (len(a)+len(b)),
                                                   p    = [float(len(a))/(len(a)+len(b)),
                                                           float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                idx_b           = np.logical_not(idx_a)
                d               = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                if np.isnan(d):
                    idx_a       = np.random.choice(a        = [0,1],
                                                   size     = (len(a)+len(b)),
                                                   p        = [float(len(a))/(len(a)+len(b)),
                                                               float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                    idx_b       = np.logical_not(idx_a)
                    d           = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                temp[n_permu,n_p] = d
        if one_tail:
            ps = (np.sum(temp >= difference,axis=0)+1.) / (n_permutation + 1.)
        else:
            ps = (np.sum(np.abs(temp) >= np.abs(difference),axis=0)+1.) / (n_permutation + 1.)
        return ps
################################################################
### multi-comparison method, reference from internet!!!!!!!!!!!!################
    ################################################
import statsmodels as sms
class MCPConverter(object):
    """
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores=None):
        self.pvals = pvals
        self.zscores = zscores
        self.len = len(pvals)
        if zscores is not None:
            srted = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals = srted[:, 0]
            self.sorted_zscores = srted[:, 1]
        else:
            self.sorted_pvals = np.array(sorted(pvals.copy()))
        self.order = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method="holm"):
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method is "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method is "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method is "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method is "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods=["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method is not "lfdr":
                    df[method] = self.adjust(method)
        return df

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
################## from https://github.com/parrt/random-forest-importances/blob/master/src/rfpimp.py#L237 #############
################## reference http://explained.ai/rf-importance/index.html #############################################
def oob_classifier_accuracy(rf, X_train, y_train):
    """
    Adjusted... 
    Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    try:
        X                   = X_train.values
    except:
        X                   = X_train.copy()
    try:
        y                   = y_train.values
    except: 
        y                   = y_train.copy()

    n_samples               = len(X)
    n_classes               = len(np.unique(y))
    # preallocation
    predictions             = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_: # for each decision tree in the random forest - I have put 1 tree in the forest
        # Private function used to _parallel_build_trees function.
        unsampled_indices   = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds          = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds

    predicted_class_indexes = np.argmax(predictions, axis=1)# threshold the probabilistic predictions
    predicted_classes       = [rf.classes_[i] for i in predicted_class_indexes] # use the thresholded indicies to obtain a binary prediction

    oob_score               = sum(y==predicted_classes) / float(len(y))
    return oob_score
def sample(X_valid, y_valid, n_samples):
    """
    Not sure what this is doing
    Only if the n_sample is less than the total number of samples, it subsamples the data???? Maybe?
    """
    if n_samples < 0: 
        n_samples                   = len(X_valid)
    n_samples                       = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix                          = np.random.choice(len(X_valid), n_samples)
        X_valid                     = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid                     = y_valid.iloc[ix].copy(deep=False)
    return X_valid, y_valid
def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=5000):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    X_sample, y_sample          = shuffle(X_train, y_train)
    # get a baseline out-of-bag sampled decoding score
    try:
        baseline                    = metric(rf, X_sample, y_sample)
    except:
        baseline                    = metric(y_sample,rf.predict_proba(X_sample)[:,-1])
    # make suer that we work on the copy of the raw data
    X_train                     = X_sample.copy() # shallow copy
    X_train                     = pd.DataFrame(X_train)
    y_train                     = y_sample
    imp                         = []
#    for n_ in range(100):
#        imp_temp = []
#        for col in X_train.columns: # for each feature
#            save                = X_train[col].copy() # save the original
#            X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
#            # oob score after reorder 1 and only 1 feature. 
#            # In orther words, how much information is gone when the feature becomes unimformative
#            try:
#                m                   = metric(rf, X_train, y_train)
#            except:
#                m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
#            X_train[col]        = save # restore the feature
#            imp_temp.append(baseline - m)
#        imp.append(imp_temp)
    
    for col in X_train.columns: # for each feature
        save                = X_train[col].copy() # save the original
        X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
        # oob score after reorder 1 and only 1 feature. 
        # In orther words, how much information is gone when the feature becomes unimformative
        try:
            m                   = metric(rf, X_train, y_train)
        except:
            m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
        X_train[col]        = save # restore the feature
        imp.append(baseline - m)
        
    return np.array(imp)#.mean(0)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def permutation_importances(rf, X_train, y_train, metric, n_samples = 5000,feature_names = None):
    """
    Call the function, and just to make it pretty
    """
    imp = permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
    imp = softmax(imp)
    I = pd.DataFrame(data={'Feature':feature_names, 'Importance':imp})
    I = I.set_index('Feature')
    return I



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def make_clfs():
    """
    Generate a dictionary with initialized scikit-learn models with their names
    1. Logistic Regression
    2. Decision Tree
    """
    return dict(
        # Sklearn applies automatic regularization, 
        # so we’ll set the parameter C to a large value to emulate no regularization
        LogisticRegression      = LogisticRegression(  C                      = 1e10,
                                                       max_iter               = int(1e3),# default was 100
                                                       tol                    = 0.0001, # default value
                                                       solver                 = 'liblinear', # default solver
                                                       random_state           = 12345
                                                        ),# end of logistic
        # I need to use random forest but with 1 tree because I use an external function to estimate the 
        # feature importance.
        RandomForestClassifier  = RandomForestClassifier(n_estimators         = 500, 
                                                         criterion            = 'entropy',
                                                         max_depth            = 1,
                                                         min_samples_leaf     = 5, # control for minimum sample per node
                                                         random_state         = 12345,
                                                         oob_score            = True,
                                                         n_jobs               = -1,
                                                         )# end of Random Forest
        )# end of dictionary
def classification(df_,
                   feature_names,
                   target_name,
                   results,
                   participant,
                   experiment,
                   window = 0,
                   n_splits = 100,
                   chance = False,
                   ):
    """
    End-to-end cross validation classification:
        Component 1: block selection
        Component 2: feature modification, target modification (mismatching trials)
        Component 3: cross validation
        Component 4: load naive classification model off-the-shelf
        Component 5: save results
    
    Inputs:
        df_                 : dataframe of a given subject
        feature_names       : names of features
        target_name         : name of target
        results             : dictionary like object, update itself every cross validation
        participant         : string, for updating the results dictionary
        experiment          : for graph of the tree model
        window              : integer value, for updating the results dictionary
        n_splits            : number of cross validation folds
        chance              : whether to estiamte the experimental score or empirical chance level
    return:
        results
    """
    features, targets = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        """
        Important note:
        the following shifting is not to directly perform the mismatching between the feature trials and
        the target trials, but it is to delete the last N rows of the feature trials and the first N rows
        of the target trials. In such case, when we put them together, the mismatching would be exactly 
        what we want: first trial of feature to predict the next trial of target.
        """
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features          = np.concatenate(features)
    targets           = np.concatenate(targets)
    if chance:
        # to shuffle the features independent from the targets
        features         = shuffle(features)
    else:
        features,targets = shuffle(features,targets)
    cv                = StratifiedShuffleSplit(
             n_splits = n_splits,# random partition for NN times
            test_size = 0.2,# split 20 % of the data as test data and the rest will be training data
         random_state = 12345 # for reproducible purpose
         )
    # this is not for initialization but for iteration
    clfs              = make_clfs()
    # for each classifier, we fit-test cross validate the classifier and save the
    # classification score and the weights of the attributes
    for model,c in clfs.items():
        scores        = []
        weights       = []
        for fold, (train,test) in enumerate(cv.split(features,targets)):
            
            clf         = make_clfs()[model]
            # fit the training data
            clf.fit(features[train],targets[train].ravel())
            # make predictions on the test data
            pred      = clf.predict_proba(features[test])[:,-1]
            # score the predictions
            score     = roc_auc_score(targets[test],pred)
            scores.append(score)
            try:
                weights.append(clf.coef_[0])
            except:
                feature_importance = permutation_importances(clf,features[train],targets[train],roc_auc_score)#clf.feature_importances_#
                weights.append(feature_importance.values.flatten())
        results['sub'       ].append(participant          )
        results['model'     ].append(model                )
        results['score'     ].append(np.mean(scores)      )
        results['window'    ].append(window               )
        for iii,name in enumerate(feature_names):
            results[name    ].append(np.mean(weights,0)[iii])
        
        print('sub {},model {},window {},score={:.3f}'.format(
                participant,
                model,
                window,
                np.mean(scores)
                )
            )
    return results
def classification_simple_logistic(df_,
                   feature_names,
                   target_name,
                   results,
                   participant,
                   experiment,
                   window = 0,
                   n_splits = 100,
                   chance = False,
                   model_name = 'LogisticRegression',
                   ):
    """
    Since the classification is redundent after the features and targets are 
    ready, I would rather to make a function for the redundent part
    
    Inputs:
        df_                 : dataframe of a given subject
        feature_names       : names of features
        target_name         : name of target
        results             : dictionary like object, update itself every cross validation
        participant         : string, for updating the results dictionary
        experiment          : for graph of the tree model
        window              : integer value, for updating the results dictionary
        n_splits            : number of cross validation folds
        chance              : whether to estiamte the experimental score or empirical chance level
    return:
        results
    """
    if len(feature_names) == 3:
        all_or_one = 'all'
    elif len(feature_names) == 1:
        all_or_one = feature_names[0]
    features, targets = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features          = np.concatenate(features)
    targets           = np.concatenate(targets)
    if chance:
        # to shuffle the features independent from the targets
        features         = shuffle(features)
    else:
        features,targets = shuffle(features,targets)
    cv                = StratifiedShuffleSplit(
             n_splits = n_splits,# random partition for NN times
            test_size = 0.2,# split 20 % of the data as test data and the rest will be training data
         random_state = 12345 # for reproducible purpose
         )
    # for each classifier, we fit-test cross validate the classifier and save the
    # classification score and the weights of the attributes
    scores        = []
    weights       = []
    for fold, (train,test) in tqdm(enumerate(cv.split(features,targets))):
        
        clf         = make_clfs()[model_name]
        # fit the training data
        clf.fit(features[train],targets[train].ravel())
        # make predictions on the test data
        pred      = clf.predict_proba(features[test])[:,-1]
        # score the predictions
        score     = roc_auc_score(targets[test],pred)
        scores.append(score)
        try:
            weights.append(clf.coef_[0])
        except:
            feature_importance = permutation_importances(clf,features[train],targets[train],roc_auc_score)#clf.feature_importances_#
            weights.append(feature_importance.values.flatten())
    results['sub'       ].append(participant          )
    results['model'     ].append(model_name           )
    results['score'     ].append(np.mean(scores)      )
    results['window'    ].append(window               )
    results['chance'].append(chance)
    results['feature'].append(all_or_one)
    
    print('sub {},model {},window {},chance = {},score={:.3f}'.format(
            participant,
            model_name,
            window,
            chance,
            np.mean(scores)
            )
        )
    return results
def correlations(df_,
                 feature_names,
                 target_name,
                 results,
                 participant,
                 experiment,
                 window = 0,
                 n_splits = 100,
                 chance = False):
    
    features, targets = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        """
        Important note:
        the following shifting is not to directly perform the mismatching between the feature trials and
        the target trials, but it is to delete the last N rows of the feature trials and the first N rows
        of the target trials. In such case, when we put them together, the mismatching would be exactly 
        what we want: first trial of feature to predict the next trial of target.
        """
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features          = np.concatenate(features)
    targets           = np.concatenate(targets)
    if chance:
        features = shuffle(features)
#        targets = shuffle(targets)
    else:
        features,targets = shuffle(features,targets)
    
    cv                = StratifiedShuffleSplit(
             n_splits = n_splits,# random partition for NN times
            test_size = 0.2,# split 20 % of the data as test data and the rest will be training data
         random_state = 12345 # for reproducible purpose
         )
    for fold, (train,test) in tqdm(enumerate(cv.split(features,targets))):
        idx_feature_names = np.arange(len(feature_names))
        for idx_feature_names_1,idx_feature_names_2 in combinations(idx_feature_names,2):
            X = features[train][:,idx_feature_names_1]
            Y = features[train][:,idx_feature_names_2]
#            r,pval = stats.pointbiserialr(X,Y)
            r = matthews_corrcoef(X,Y)
            paval = 1.
            results['sub'           ].append(participant            )
            results['correlation'   ].append(r                      )
            results['pvals'         ].append(pval                   )
            results['window'        ].append(window                 )
            results['fold'          ].append(fold                   )
            results['feature_name_1'].append(feature_names[idx_feature_names_1])
            results['feature_name_2'].append(feature_names[idx_feature_names_2])
        
    
    
    return results

def cv_counts(df_,
              feature_names,
              target_name,
              results,
              participant,
              experiment,
              window = 0,
              n_splits = 100,):
    
    features, targets = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        """
        Important note:
        the following shifting is not to directly perform the mismatching between the feature trials and
        the target trials, but it is to delete the last N rows of the feature trials and the first N rows
        of the target trials. In such case, when we put them together, the mismatching would be exactly 
        what we want: first trial of feature to predict the next trial of target.
        """
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features          = np.concatenate(features)
    targets           = np.concatenate(targets)
    
    features,targets = shuffle(features,targets)
    cv                = StratifiedShuffleSplit(
             n_splits = n_splits,# random partition for NN times
            test_size = 0.2,# split 20 % of the data as test data and the rest will be training data
         random_state = 12345 # for reproducible purpose
         )
    for fold, (train,test) in tqdm(enumerate(cv.split(features,targets))):
        y = targets[train].astype(int)
        X = features[train].astype(int)
        
        df_temp = pd.DataFrame(X,columns=feature_names)
        df_temp[target_name] = y
        
        for name in feature_names:
            results['{}_high_cond_{}_low'.format(target_name,name)].append(
                    np.sum(np.logical_and(df_temp[name] == 0,# low conditioner
                                          df_temp[target_name] == 1)) / np.sum(df_temp[name] == 0))
            results['{}_high_cond_{}_high'.format(target_name,name)].append(
                    np.sum(np.logical_and(df_temp[name] == 1, # high conditioner
                                          df_temp[target_name] == 1)) / np.sum(df_temp[name] == 1))
            
            
            
        results['sub'           ].append(participant            )
        results['window'        ].append(window                 )
        results['fold'          ].append(fold                   )
        
    return results

def predict(instances, weights,intercept):
    weights = weights.reshape(-1,1)
    """Predict gender given weight (w) and height (h) values."""
    v = intercept + np.dot(instances,weights)
    return np.exp(v)/(1+np.exp(v))
def bayesian_logistic(df_,
                   feature_names,
                   target_name,
                   results,
                   participant,
                   experiment,
                   dot_dir,
                   window=0,
                   ):
    """
    Since the classification is redundent after the features and targets are 
    ready, I would rather to make a function for the redundent part
    
    Inputs:
        df_                 : dataframe of a given subject
        feature_names       : names of features
        target_name         : name of target
        results             : dictionary like object, update itself every cross validation
        participant         : string, for updating the results dictionary
        experiment          : for graph of the tree model
        dot_dit             : directory of the tree plots
        window              : integer value, for updating the results dictionary
    return:
        results
    """
    features, targets   = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features            = np.concatenate(features)
    targets             = np.concatenate(targets)
    features, targets   = shuffle(features,targets)
    # for each classifier, we fit-test cross validate the classifier and save the
    # classification score and the weights of the attributes
    model_name               = 'bayes_logistic'
   
    # this is for initialization
    df_train        = pd.DataFrame(features,columns=feature_names)
    df_train[target_name]   = targets
    scaler = StandardScaler()
    
    for name in feature_names:
       if 'RT' in name:
           df_train[name] = scaler.fit_transform(df_train[name].values.reshape(-1,1))
    niter                   = 1000
    formula                 = '{} ~'.format(target_name)
    for name in feature_names:
        formula += ' + {}'.format(name)
    
    with pm.Model() as model:
        pm.glm.GLM.from_formula(formula, 
                                df_train, 
                                family=pm.glm.families.Binomial(),
                                )
        start               = pm.find_MAP(progressbar=False)
        try:
            step                = pm.NUTS(scaling=start,)
        except:
            step       = pm.Metropolis()
        trace               = pm.sample(  niter, 
                                          start         =start, 
                                          step          =step, 
                                          njobs         =4, 
                                          random_seed   =12345,
                                          progressbar=0,
                                          )
    df_trace                = pm.trace_to_dataframe(trace[niter//2:])
    intercept               = df_trace['Intercept'].mean()
    df_test                 = pd.DataFrame(features,
                                           columns      =feature_names)
    weights = df_trace.iloc[:,1:].values.mean(0)
    preds                   = predict(df_test.values,weights,intercept)
    # score the predictions
    score                   = roc_auc_score(targets,preds)
    results['sub'       ].append(participant          )
    results['model'     ].append(model_name                )
    results['score'     ].append(score      )
    results['window'    ].append(window               )
    for iii,name in enumerate(feature_names):
        results[name        ].append(df_trace[name].values.mean())
    
    print('sub {},model {},window {},score={:.3f}'.format(
            participant,
            model_name,
            window,
            score
            )
        )
    return results,trace
def post_processing(df,id_vars=[],value_vars=[]):
    """
    After classification, we want to see how each model weights on the attributes,
    thus, we need some ways to nornalize the weights for each subject so that 
    we could have a fair comparison between subjects and models
    Here, I decide to normize the vector of 3 attributes for each subject so that
    all these weight vectors are norm vecoters. In other words, the weights can
    be interpret as the directions they are pointing in the 3D space
    """
    df_ = df.copy()
    for feature_name in value_vars:
        df_[feature_name][df_['model'] == 'LogisticRegression'] = df_[feature_name][df['model'] == 'LogisticRegression'].apply(np.exp)
    df_post               = pd.melt(df_,
                                    id_vars         =id_vars,
                                    value_vars      =value_vars)
    df_post.columns       = ['Models',
                             'Scores',
                             'sub',
                             'Window',
                             'Attributes',
                             'Values',
                             ]
    return df_post
def post_processing2(df,names = []):
    """

    """
    # To 
    temp1                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[0]
                                ]]
    temp1.loc[:,'value']  = temp1[names[0]]
    temp1['Attributions'] = names[0]
    temp2                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[1]
                                ]]
    temp2.loc[:,'value']  = temp2[names[1]]
    temp2['Attributions'] = names[1]
    temp3                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[2]
                                ]]
    temp3.loc[:,'value']  = temp3[names[2]]
    temp3['Attributions'] = names[2]
    df                    = pd.concat([temp1,
                                       temp2,
                                       temp3]).dropna(axis=1)
    return df

def multiple_pairwise_comparison(df,n_ps=100,n_permutation=5000,method='bonferoni'):
    from itertools import combinations
    results = dict(model        =[],
                   window       =[],
                   ps_mean      =[],
                   ps_std       =[],
                   larger       =[],
                   less         =[],
                   diff         =[],
                   )
    for (model,window), df_sub in df.groupby(['model','window']):
        name = list(df_sub.columns[:6])
        names = combinations(name,2)
        for (name1,name2) in names:

                if np.mean(df_sub[name1] - df_sub[name2]) >= 0:
                    ps = resample_ttest_2sample(df_sub[name1].values,
                                        df_sub[name2].values,
                                        n_ps=n_ps,
                                        n_permutation=n_permutation)
                    results['model'].append(model)
                    results['window'].append(window)
                    results['ps_mean'].append(ps.mean())
                    results['ps_std'].append(ps.std())
                    results['larger'].append(name1)
                    results['less'].append(name2)
                    results['diff'].append(np.mean(df_sub[name1] - df_sub[name2]))
                else:
                    ps = resample_ttest_2sample(df_sub[name2].values,
                                        df_sub[name1].values,
                                        n_ps=n_ps,
                                        n_permutation=n_permutation)
                    results['model'].append(model)
                    results['window'].append(window)
                    results['ps_mean'].append(ps.mean())
                    results['ps_std'].append(ps.std())
                    results['larger'].append(name2)
                    results['less'].append(name1)
                    results['diff'].append(np.mean(df_sub[name2] - df_sub[name1]))
    compar = pd.DataFrame(results)
    temp = []
    for (model,window),compar_sub in compar.groupby(['model','window']):
        idx_sort = np.argsort(compar_sub['ps_mean'])
        for name in compar_sub.columns:
            compar_sub[name] = compar_sub[name].values[idx_sort]
        convert = MCPConverter(compar_sub['ps_mean'].values)
        df_pvals = convert.adjust_many()
        compar_sub['ps_corrected'] = df_pvals[method].values
        temp.append(compar_sub)
    compar = pd.concat(temp)
    return compar
#def tinvlogit(x):
#    return t.exp(x) / (1 + t.exp(x))
def logistic_regression(df_working,sample_size=3000):
    independent_variables = df_working.columns[:-1]
    dependent_variable = df_working.columns[-1]
    traces,models = {},{}
    for name in independent_variables:
        with pm.Model() as model:
            pm.glm.GLM.from_formula('{} ~ {}'.format(dependent_variable,name),
                                       df_working, family=pm.glm.families.Binomial())
            start = pm.find_MAP()
            step = pm.NUTS(scaling=start)
            trace = pm.sample(sample_size,step,start,chains=3, tune=1000)
        traces[name]=trace
        models[name]=model
    return traces,models
def compute_r2(df, ppc, ft_endog):
    
    sse_model = (ppc['y'] - df[ft_endog].values)**2
    sse_mean = (df[ft_endog].values - np.random.choice(df[ft_endog],
                size=(ppc['y'].shape[0],df[ft_endog].shape[0])
                ))**2
    
    return 1 - (sse_model.sum(1) / sse_mean.sum(1))
def plot_traces(traces, retain=1000):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')
def compute_ppc(trc, mdl, samples=500, size=50):
    return pm.sample_ppc(trc[-1000:], samples=samples, model=mdl,)
def compute_mse(df, ppc, ft_endog):
    return np.sum((ppc['y'].mean(0).mean(0).T - df[[ft_endog]])**2)[0]/df.shape[0]
def classification_leave_one_sub_out(features,targets,results,groups,experiment,dot_dir,window=0,comparison='fast'):
    from sklearn.preprocessing import normalize
    features,targets,groups = shuffle(features,targets,groups)
    cv = LeaveOneGroupOut()
    clfs = make_clfs()
    
    for model,c in clfs.items():
        scores = []
        weights = []
        print('\ntrain-test')
        for fold,(train,test) in enumerate(cv.split(features,targets,groups=groups)):
            clf = make_clfs()[model]
            clf.fit(features[train],targets[train].ravel())
            pred = clf.predict_proba(features[test])[:,-1]
            score = roc_auc_score(targets[test],pred)
            scores.append(score)
            try:
                weights.append(normalize(clf.coef_)[0])
            except:
                weights.append(normalize(clf.feature_importances_.reshape(1, -1))[0])
        results['model'     ].append(model                )
        results['score'     ].append(np.mean(scores)      )
        results['window'    ].append(window               )
        results['correct'   ].append(np.mean(weights,0)[0])
        results['awareness' ].append(np.mean(weights,0)[1])
        results['confidence'].append(np.mean(weights,0)[2])
        # test against to chance level
        print('estimate chance level')
        if comparison == 'slow':
            random_score = []
            for _ in tqdm(range(5)):
                clf = make_clfs()[model]
                cv  = LeaveOneGroupOut()
                X_  = features
                y_  = shuffle(targets)
                random_score_ = cross_val_score(clf,X_,y_.ravel(),groups=groups,
                                               scoring='roc_auc',cv=cv)
                random_score.append(random_score_)
            random_score = np.mean(random_score,axis=0)
            ps = resample_ttest_2sample(np.array(scores),random_score,
                                        n_ps=500,n_permutation=10000)
            results['p_val'     ].append(np.mean(ps))
        elif comparison == 'fast':
            clf = make_clfs()[model]
            cv  = LeaveOneGroupOut()
            X_  = features
            y_  = shuffle(targets)
            random_score = cross_val_score(clf,X_,y_.ravel(),groups=groups,
                                           scoring='roc_auc',cv=cv)
            ps = resample_ttest_2sample(np.array(scores),random_score,
                                        n_ps=500,n_permutation=10000)
            results['p_val'     ].append(np.mean(ps))
        if model == 'DecisionTreeClassifier':
            clf.fit(features,targets)
            out_file = dot_dir+'/'+'{}_window_{}_LOG_tree.dot'.format(experiment,window)
            export_graphviz(decision_tree=clf,
                            out_file=out_file,
                            feature_names=['correct',
                                           'awareness',
                                           'confidence'],
                            class_names=['low {}'.format(experiment),
                                         'high {}'.format(experiment)])
        print('model {},window {},score={:.3f}-{:.4f}'.format(
                model,
                window,
                np.mean(scores),
                np.mean(ps)
                )
            )
    return results
from matplotlib import pyplot as plt
def errplot(x, y, yerr, **kwargs):
    ax      = plt.gca()
    data    = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse             = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
def preprocessing(df,
                  participant,
                  n_back                    = 0,
                  independent_variables     = [],
                  dependent_variable        = [],):
    features = []
    targets  = []
    for block,df_block in df.groupby('blocks'):
        df_block['row'] = np.arange(df_block.shape[0])
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        feature       = (df_block[independent_variables].shift(n_back) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[dependent_variable].shift(-n_back) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features    = np.concatenate(features)
    targets     = np.concatenate(targets)
    subs        = np.array([participant] * len(targets))
    return features,targets,subs

def get_features_targets_groups(df,
                                n_back                  = 0,
                                independent_variables   = [],
                                dependent_variable      = []):
    X,y,groups                                          = [],[],[]
    
    for participant,pos_sub in df.groupby('participant'):# for each subject
        features,targets,subs = preprocessing(pos_sub,
                                              participant,
                                              n_back                = n_back,
                                              independent_variables = independent_variables,
                                              dependent_variable    = dependent_variable)
        X.append(features)
        y.append(targets)
        groups.append(subs)
    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)
    return X,y,groups
def posthoc_multiple_comparison(df_sub,
                                depvar = '',
                                factor='',
                                n_ps=100,
                                n_permutation=5000):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            level1 = [],
            level2 = [],
            mean1 = [],
            mean2 = [],
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='pairs')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[df_sub[factor] == level1]
        b = df_sub[df_sub[factor] == level2]
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
            a = df_sub[df_sub[factor] == level1]
            b = df_sub[df_sub[factor] == level2]
        ps = resample_ttest_2sample(a[depvar].values,
                                    b[depvar].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['level1'].append(level1)
        results['level2'].append(level2)
        results['mean1'].append(a[depvar].values.mean())
        results['mean2'].append(b[depvar].values.mean())
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_scipy(df_sub,depvar = '',factor='',):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='pairs')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
        b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
            a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
            b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        t,pval = stats.ttest_rel(a[depvar].values,
                                 b[depvar].values,)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)*2-2)
        results['level1'].append(level1)
        results['level2'].append(level2)
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction(df_sub,
                                            depvar = '',
                                            unique_levels = [],
                                            n_ps=100,
                                            n_permutation=5000,
                                            selected = 0):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            level1 = [],
            level2 = []
            )
    unique_factor1 = np.unique(df_sub[unique_levels[0]])
    unique_factor2 = np.unique(df_sub[unique_levels[1]])
    pairs = [[a,b] for a in unique_factor1 for b in unique_factor2]
    pairs = combinations(pairs,2)
    if selected is not None:
        temp = []
        for pair in pairs:
            level1 = pair[0]
            level2 = pair[1]
            if level1[selected] == level2[selected]:
                temp.append((level1,level2))
        pairs = temp
    else:
        pass
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        ps = resample_ttest_2sample(a[depvar].values,
                                    b[depvar].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['level1'].append('{}_{}'.format(level1[0],level1[1]))
        results['level2'].append('{}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction_scipy(df_sub,
                                            depvar = '',
                                            unique_levels = [],
                                            ):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            level1 = [],
            level2 = []
            )
    unique_factor1 = np.unique(df_sub[unique_levels[0]])
    unique_factor2 = np.unique(df_sub[unique_levels[1]])
    pairs = [[a,b] for a in unique_factor1 for b in unique_factor2]
    pairs = combinations(pairs,2)
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        t,pval = stats.ttest_rel(a[depvar].values,
                                 b[depvar].values,
                                    )
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(a)+len(b) - 2)
        results['level1'].append('{}_{}'.format(level1[0],level1[1]))
        results['level2'].append('{}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results

def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'

def strip_interaction_names(df_corrected):
    results = []
    for ii,row in df_corrected.iterrows():
        row['window'] = row['level1'].split('_')[0]
        row['attribute1']= row['level1'].split('_')[1]
        row['attribute2']= row['level2'].split('_')[1]
        results.append(row.to_frame().T)
    results = pd.concat(results)
    return results
def compute_xy(df_sub,position_map,hue_map):
    df_add = []
    for ii,row in df_sub.iterrows():
        xtick = int(row['window']) - 1
        attribute1_x = xtick + position_map[hue_map[row['attribute1']]]
        attribute2_x = xtick + position_map[hue_map[row['attribute2']]]
        row['x1'] = attribute1_x
        row['x2'] = attribute2_x
        df_add.append(row.to_frame().T)
    df_add = pd.concat(df_add)
    return df_add
