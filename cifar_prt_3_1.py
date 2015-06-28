# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:50:31 2015
@author: dsing001
"""

from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn import datasets 
import math
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import NearestNeighbors as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import random 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as rfc
imgsize = 32
prt = 1
dgts_data = pd.read_csv("cifar.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape
dgts_data = np.array(dgts_data)
print dgts_data.shape
#print dgts_data

#add 16 by 16 as well
dt_data = np.reshape(dgts_data,(dgts_data.shape[0],imgsize,imgsize))
for i in range(imgsize-prt):
    rw1 = i
    rw2 = i + prt
    for j in range(imgsize-prt):
        col1 = j
        col2 = j + prt
        ar = np.reshape(dt_data[:,rw1:rw2,col1:col2],(dt_data.shape[0],prt*prt))
        ad= np.reshape(np.mean(ar,axis=1),(dt_data.shape[0],1))
        dgts_data = np.hstack((dgts_data,ad))
      
print dgts_data.shape
rmsize = imgsize*imgsize
dgts_data = dgts_data[:,rmsize:]
print dgts_data.shape


n_neighbours = range(7,8,2)
print n_neighbours


dgts_lbl = pd.read_csv("cifar_l.csv",index_col=0)

lbls = np.array(dgts_lbl)
print lbls.shape

#train_lbl,test_lbl,train_dt,test_dt = train_test_split(lbls,dt,test_size = 0.15,random_state=1299004)
train_class = lbls
train_data = dgts_data

dt = pd.read_csv("cifar_test.csv",index_col=0)

dt = np.array(dt)
print dt.shape
dgts_data = dt
dt_data = np.reshape(dt,(dt.shape[0],imgsize,imgsize))
for i in range(imgsize-prt):
    rw1 = i
    rw2 = i + prt
    for j in range(imgsize-prt):
        col1 = j
        col2 = j + prt
        ar = np.reshape(dt_data[:,rw1:rw2,col1:col2],(dt_data.shape[0],prt*prt))
        ad= np.reshape(np.mean(ar,axis=1),(dt_data.shape[0],1))
        dgts_data = np.hstack((dgts_data,ad))
      


lbls = pd.read_csv("cifar_test_l.csv",index_col=0)

lbls = np.array(lbls)
print lbls.shape

test_class = lbls
dgts_data = dgts_data[:,rmsize:]
test_data = dgts_data


for nb in n_neighbours:
    print nb
    mdl = knn(n_neighbors= nb)
    mdl.fit(train_data,train_class)
    print 'score'
    print mdl.score(test_data,test_class)
    print 
    #print mdl.feature_importances_





