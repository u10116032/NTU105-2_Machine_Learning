# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:37:10 2017

@author: halley
"""
import pandas 
from CFModel import CFModel
import numpy as np
import sys


test_data = pandas.read_csv(sys.argv[1] + 'test.csv', sep=',')
test_userid = np.array(test_data['UserID'], dtype= np.float64)
test_movieid = np.array(test_data['MovieID'], dtype= np.float64)

columns = ['TestDataID','Rating']
df = pandas.DataFrame(columns=columns)

model = CFModel(6040+1, 3952+1, 200)
model.load_weights('./MF_Model/model_MF.h5')

print('load model finished')
#%%
result = model.predict([test_userid, test_movieid, test_userid, test_movieid]).flatten()

print('predict finish, start writing to csv.')
'''
for i in range(len(result)):   
    temp = result[i]
    if temp < 0:
        temp = 0
    if temp > 5:
        temp = 5
    df.loc[i,'TestDataID'] = str(i+1)
    df.loc[i,'Rating'] = temp
    #print("test data[",i,"]: ",temp)
df.to_csv(sys.argv[2], sep=',', index=False)
'''
with open(sys.argv[2], 'w') as file:
    file.write('TestDataID,Rating\n')
    for i in range(len(result)):
        temp = result[i]
        if temp < 0:
            temp = 0
        if temp > 5:
            temp = 5
        file.write(str(i+1)+','+str(temp)+'\n')

