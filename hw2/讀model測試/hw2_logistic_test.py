#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 00:28:30 2017

@author: halley
"""

import pandas
import sys
import numpy as np

model = pandas.read_csv("./model_logistic.csv")

b = model.loc[0, 'b']
w = model.loc[0:105, 'w'].as_matrix().astype(np.float64)
firstSix_max = model.loc[0:5, 'first_six_max'].as_matrix().astype(np.float64)
firstSix_min = model.loc[0:5, 'first_six_min'].as_matrix().astype(np.float64)

# Test Data Handling
test_data = pandas.read_csv(sys.argv[1], encoding= 'big5', header = None) # 1~16281
test_input = []
test_output = []
for i in range(1, len(test_data)):
    firstSix_normalized = test_data.loc[i].as_matrix().astype(np.float)
    firstSix_normalized[0:6] = (firstSix_normalized[0:6]-firstSix_min) / (firstSix_max - firstSix_min)
    if sigmoid(b+ np.dot(w, firstSix_normalized)) > 0.5:
        test_output.append(int(1))
    else:
        test_output.append(int(0))

    
        
#output predict data
columns = ['id','label']
df = pandas.DataFrame(columns=columns)

for i in range(len(test_output)):
    df.loc[i,'id']=str(i+1)
    df.loc[i,'label']=test_output[i]
    
df.to_csv(sys.argv[2], sep=',', index=False)  
