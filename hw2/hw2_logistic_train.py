#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:28:54 2017

@author: halley
"""
#%%
import pandas
import numpy as np
import sys

#%%
def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 0.0000001, 0.9999999)

# Training data handling
data = pandas.read_csv(sys.argv[1], encoding= "big5", header=None) # 1~32561
label_data = pandas.read_csv(sys.argv[2], encoding= 'big5', header=None,dtype= np.int) # 0~32560

x_data = []
y_data = []
firstSix_max = np.amax(data.loc[1:,0:5].as_matrix().astype(np.float64), axis= 0)
firstSix_min = np.amin(data.loc[1:,0:5].as_matrix().astype(np.float64), axis= 0)

for i in range(len(label_data)):
    firstSix_normalized = data.loc[i+1].as_matrix().astype(np.float64)
    firstSix_normalized[0:6] = (firstSix_normalized[0:6]-firstSix_min) / (firstSix_max - firstSix_min)
    x_data.append(firstSix_normalized)
    y_data.append(label_data.loc[i,0].astype(np.float64))

#%% 

model = pandas.read_csv("./initial_parameters_logistic.csv")

b = model.loc[0, 'b']
w = model.loc[0:105, 'w'].as_matrix().astype(np.float64)


# Start Training 
# y = sigmoid(b + wx )
# if y >0.5 output class2: 1
# else output class1: 0
# Loss Function : Cross Entropy

iteration = 10000
b_lr = 0.0
w_lr = 0.0
smooth_lambda = 0.75

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros((106,), dtype= np.float64)
    loss_value = np.float64(0)
    for n in range (len(x_data)):
        temp_sigmoid = sigmoid(np.dot(w,x_data[n])+b)
        temp = np.float64((-1)*(y_data[n] - temp_sigmoid))
        b_grad += temp
        w_grad += temp * x_data[n]
        loss_value += (-1)*(y_data[n] * np.log(temp_sigmoid)+(1-y_data[n]) * np.log(1- temp_sigmoid)) 
    w_grad += 2 * smooth_lambda * w[0]
    b_lr += b_grad**2
    w_lr += w_grad**2
    b = b - (1/np.sqrt(b_lr)) * b_grad
    w = w - (1/np.sqrt(w_lr)) * w_grad

    print(str((i+1)*100/iteration) + '%, Loss Value: ' + str(loss_value/len(x_data)))

    
#%%
'''
# Test Data Handling
test_data = pandas.read_csv('X_test.csv', encoding= 'big5', header = None) # 1~16281
test_input = []
test_output = []
for i in range(1, len(test_data)):
    firstSix_normalized = test_data.loc[i].as_matrix().astype(np.float)
    firstSix_normalized[0:6] = (firstSix_normalized[0:6]-firstSix_min) / (firstSix_max - firstSix_min)
    test_input.append(firstSix_normalized)
    
# predict test data
for elm in test_input:
    if sigmoid(b+ np.dot(w, elm)) > 0.5:
        test_output.append(int(1))
    else:
        test_output.append(int(0))
        
#output predict data
columns = ['id','label']
df = pandas.DataFrame(columns=columns)

for i in range(len(test_output)):
    df.loc[i,'id']=str(i+1)
    df.loc[i,'label']=test_output[i]
    
df.to_csv('predict_logistic.csv', sep=',', index=False)  
'''

# save model parameters
columns = ['b','w']
df = pandas.DataFrame(columns=columns)

df.loc[0,'b']=b
for i in range(106):
    df.loc[i,'w']=w[0,i]
    #df.loc[i,'a']=a[0,i]
    
df.to_csv('model_logistic.csv', sep=',', index=False)