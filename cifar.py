
"""
Created on Sat Jun 27 13:23:22 2015

@author: Inpiron
"""
import numpy as np
import cPickle
import pandas as pd
from matplotlib import pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
def unpickle(file):    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
fl = 'data_batch_1'

ab = unpickle(fl)
print ab
dt = ab['data']
lbls = ab['labels']
lbls = np.array(lbls).reshape(dt.shape[0],1)

print dt.shape
dt = np.array(dt).reshape(dt.shape[0],1024,3)
dt_1 = dt[2,:,0].reshape(32,32)
print dt_1
plt.imshow(dt_1)
plt.show()
print ghyyyy
print dt.shape
dt = rgb2gray(dt)
print dt.shape
print dt

fl = 'data_batch_2'
ab = unpickle(fl)
dt1 = ab['data']
lbls1 = ab['labels']
lbls1 = np.array(lbls1).reshape(dt1.shape[0],1)
print dt.shape
dt1 = np.array(dt1).reshape(dt1.shape[0],1024,3)
print dt1.shape
dt1 = rgb2gray(dt1)
print dt1.shape
print dt1
dt = np.vstack((dt,dt1))
lbls = np.vstack((lbls,lbls1))
print lbls.shape

fl = 'data_batch_3'
ab = unpickle(fl)
dt1 = ab['data']
lbls1 = ab['labels']
lbls1 = np.array(lbls1).reshape(dt1.shape[0],1)
print dt.shape
dt1 = np.array(dt1).reshape(dt1.shape[0],1024,3)
print dt1.shape
dt1 = rgb2gray(dt1)
print dt1.shape
print dt1
dt = np.vstack((dt,dt1))
lbls = np.vstack((lbls,lbls1))
print lbls.shape


fl = 'data_batch_4'
ab = unpickle(fl)
dt1 = ab['data']
lbls1 = ab['labels']
lbls1 = np.array(lbls1).reshape(dt1.shape[0],1)
print dt.shape
dt1 = np.array(dt1).reshape(dt1.shape[0],1024,3)
print dt1.shape
dt1 = rgb2gray(dt1)
print dt1.shape
print dt1
dt = np.vstack((dt,dt1))
lbls = np.vstack((lbls,lbls1))
print lbls.shape

fl = 'data_batch_5'
ab = unpickle(fl)
dt1 = ab['data']
lbls1 = ab['labels']
lbls1 = np.array(lbls1).reshape(dt1.shape[0],1)
print dt.shape
dt1 = np.array(dt1).reshape(dt1.shape[0],1024,3)
print dt1.shape
dt1 = rgb2gray(dt1)
print dt1.shape
print dt1
dt = np.vstack((dt,dt1))
lbls = np.vstack((lbls,lbls1))
print lbls.shape

dt = pd.DataFrame(dt)
dt.to_csv('cifar.csv')
lbls = pd.DataFrame(lbls)
lbls.to_csv('cifar_l.csv')


    
fl = 'test_batch'

ab = unpickle(fl)
dt = ab['data']
lbls = ab['labels']
lbls = np.array(lbls).reshape(dt.shape[0],1)

print dt.shape
dt = np.array(dt).reshape(dt.shape[0],1024,3)
print dt.shape
dt = rgb2gray(dt)
print dt.shape
print dt

dt = pd.DataFrame(dt)
dt.to_csv('cifar_test.csv')
lbls = pd.DataFrame(lbls)
lbls.to_csv('cifar_test_l.csv')


