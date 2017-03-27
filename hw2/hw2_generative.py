#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:11:43 2017

@author: halley
"""
#%%
import pandas
import numpy as np
import sys
#%%
#calculate post probabilities
def average(list):
    sum = np.zeros((106, ))
    for elm in list:
        sum += elm
    return sum/len(list)

def variance_matrix(u_data, u):
    sum = np.zeros((len(u), len(u)))
    for elm in u_data:
        temp = np.transpose(np.matrix(elm)) - u
        sum += np.matmul( temp , np.transpose(temp) )
    return sum/len(u_data)

def sigmoid(z):
    return 1/(1+np.exp(-z))
    

data = pandas.read_csv(sys.argv[1], encoding= "big5", header=None) # 1~32561
label_data = pandas.read_csv(sys.argv[2], encoding= 'big5', header=None,dtype= np.int) # 0~32560

u1_data = [] #  <  50K
u2_data = [] #  >= 50K

for i in range(len(label_data)):
    if label_data.loc[i,0]:          # >= 50K
        u2_data.append(data.loc[i+1].as_matrix().astype(np.float))
    else:                            # < 50K
        u1_data.append(data.loc[i+1].as_matrix().astype(np.float)) 

u1 = np.transpose(np.matrix(average(u1_data)))
u2 = np.transpose(np.matrix(average(u2_data)))

var1 = variance_matrix(u1_data, u1)
var2 = variance_matrix(u2_data, u2)

inv＿cvm = np.linalg.inv((len(u1_data)/len(label_data))*var1 + (len(u2_data)/len(label_data))*var2)

w = np.transpose(u1 - u2)* inv_cvm
b = (-1/2)*np.transpose(u1) * inv_cvm * u1 + (1/2)*np.transpose(u2) * inv_cvm * u2 + np.log(len(u1_data)/len(u2_data))

#%%
# test data handling
test_data = pandas.read_csv(sys.argv[3], encoding= 'big5', header = None) # 1~16281
test_input = []
test_output = []
for i in range(1, len(test_data)):
    test_input.append(test_data.loc[i].as_matrix().astype(np.float))
    
# predict test data
for elm in test_input:
    if sigmoid(np.dot(w, elm)+b) > 0.5:
        test_output.append(int(0))
    else:
        test_output.append(int(1))
        
#output predict data
columns = ['id','label']
df = pandas.DataFrame(columns=columns)

for i in range(len(test_output)):
    df.loc[i,'id']=str(i+1)
    df.loc[i,'label']=test_output[i]
    
df.to_csv(sys.argv[4], sep=',', index=False)


