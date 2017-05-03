#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:49:29 2017

@author: halley
"""

import pandas
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys

#%%
x_test = []
test_data = pandas.read_csv(sys.argv[1])

for i in range(test_data.shape[0]):
    temp = np.array(list(map(int, test_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_test.append(temp)
x_test = np.array(x_test)    

#%%
model = load_model('./Final_Model_v3.h5')

predict_result = model.predict_classes(x_test, batch_size=50)
#predict_result_prob = model.predict_proba(x_test, batch_size=50)

#%%
#Output Predict Data
columns = ['id','label']
df = pandas.DataFrame(columns=columns)

for i in range(len(predict_result)):
    df.loc[i,'id']=str(i)
    df.loc[i,'label']=predict_result[i]
df.to_csv(sys.argv[2], sep=',', index=False)