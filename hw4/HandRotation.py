#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:16:14 2017

@author: halley
"""

#%%
from PIL import Image
import numpy as np

imgs = []
for i in range(481):
    img = Image.open('./hand/hand.seq'+str(i+1)+'.png')
    imgs.append(np.array(img, dtype= np.float).flatten())
imgs = np.array(imgs, dtype= np.float)      
imgs_var = np.var(np.var(imgs, axis=1), axis=0)

#%%

import numpy as np
from sklearn.cluster import KMeans

#find the index of the nearest center
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx+1

#calculate each variance of set
data= np.load('data.npz')
var = []
for i in data.keys():
    elm = data[i].flatten().astype(np.float)
    var.append(np.var(elm))
var = np.asarray(var).reshape(len(var), 1)


#K means algorithm to calculate each center
kmeans= KMeans(n_clusters=60, random_state=5, n_init= 50).fit(var)
centers = np.sort(kmeans.cluster_centers_, axis = 0)

#find the index of nearest center
result = []
for each_var in var:
    result.append(find_nearest(centers, each_var))
   
#%%     
print(find_nearest(centers, imgs_var))