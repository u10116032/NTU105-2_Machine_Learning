# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
from sklearn.cluster import KMeans
import pandas 
import sys

#find the index of the nearest center
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx+1

#%%

#calculate each variance of set
data= np.load(sys.argv[1])
var = []
for i in data.keys():
    elm = data[i].flatten().astype(np.float)
    var.append(np.var(elm))
var = np.asarray(var).reshape(len(var), 1)

#%%
#K means algorithm to calculate each center
kmeans= KMeans(n_clusters=60, random_state=5, n_init= 50).fit(var)
centers = np.sort(kmeans.cluster_centers_, axis = 0)

#find the index of nearest center
result = []
for each_var in var:
    result.append(find_nearest(centers, each_var))
        
#%%

#Output Predict Data
columns = ['SetId','LogDim']
df = pandas.DataFrame(columns=columns)

for i in range(len(result)):
    df.loc[i,'SetId']=i
    df.loc[i,'LogDim']=np.log(result[i])
    
df.to_csv(sys.argv[2], sep=',', index=False)






