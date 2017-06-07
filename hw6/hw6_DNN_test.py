# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:37:10 2017

@author: halley
"""
import pandas 
from CFModel import DeepModel
import numpy as np
import sys


test_data = pandas.read_csv(sys.argv[1]+'test.csv', sep=',')
test_userid = test_data['UserID']
test_movieid = test_data['MovieID']

columns = ['TestDataID','Rating']
df = pandas.DataFrame(columns=columns)

model2 = DeepModel(6040+1, 3952+1, 200)
model2.load_weights('./DNN_Model/model_DNN_2.h5')
model3 = DeepModel(6040+1, 3952+1, 200)
model3.load_weights('./DNN_Model/model_DNN_3.h5')
model5 = DeepModel(6040+1, 3952+1, 200)
model5.load_weights('./DNN_Model/model_DNN_5.h5')
model6 = DeepModel(6040+1, 3952+1, 200)
model6.load_weights('./DNN_Model/model_DNN_6.h5')
print('load model finished')

result2 = model2.predict([np.array(test_userid),np.array(test_movieid), np.array(test_userid),np.array(test_movieid)]).flatten()
result3 = model3.predict([np.array(test_userid),np.array(test_movieid), np.array(test_userid),np.array(test_movieid)]).flatten()
result5 = model5.predict([np.array(test_userid),np.array(test_movieid), np.array(test_userid),np.array(test_movieid)]).flatten()
result6 = model6.predict([np.array(test_userid),np.array(test_movieid), np.array(test_userid),np.array(test_movieid)]).flatten()

print('predict finish')
avg_result = (result2 + result3 + result5 + result6 )/4
print('start writing to csv')
for i in range(len(avg_result)):   
    temp = avg_result[i]
    if temp < 0:
        temp = 0
    if temp > 5:
        temp = 5
    df.loc[i,'TestDataID'] = str(i+1)
    df.loc[i,'Rating'] = temp
    #print("test data[",i,"]: ",temp)
df.to_csv(sys.argv[2], sep=',', index=False)

