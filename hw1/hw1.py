#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:47:20 2017

@author: halley
"""
#%%
import pandas
import numpy as np
import sys

#%%
data = pandas.read_csv(sys.argv[1], encoding="big5")

x_data = []
y_data = []

delta_window = 10 #1~10

# Training Data Handling
for i in range(0, len(data.index), 18):
    for j in range(0, 13, delta_window):
                  
        #0-8 hour per day in x_data
        temp = data.loc[i+9].loc[str(j):str(j+8)].as_matrix()
                
        temp = temp.astype(np.float)  
        x_data.append(temp)
        y_data_raw = float(data.iloc[9+i][str(j+9)])
        y_data.append(y_data_raw)
        
        


b = 2
w = np.array([0.00749081146757, 0.121930150312, 0.257802575574, 0.364190693063, 0.0312678995303, 0.305631459067, 0.98580021091, 0.59675082927, 0.00997674460044
], dtype=np.float)
a = np.array([0.00749081146757, 0.121930150312, 0.257802575574, 0.364190693063, 0.0312678995303, 0.305631459067, 0.98580021091, 0.59675082927, 0.00997674460044
], dtype=np.float)
#a = np.random.random((1,9))

"""
# save parameters
arg_data = open('arg_data.txt', 'w')
arg_data.writelines('b:'+'\n')
arg_data.writelines(str(b)+'\n')
arg_data.writelines('w:'+'\n')
for elm in w:
    arg_data.write(str(elm)+ "\n")
arg_data.close()
"""


        
                     
# Start Training
# ydata = b + w * xdata  + a * xdata^2
# w = 1*9
# xdata = 9*1

iteration = 50000

b_lr = 0.0
w_lr = 0.0
a_lr = 0.0

lambda_smooth = 0.1

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros((1 * 9, ), dtype= np.float)
    a_grad = np.zeros((1 * 9, ), dtype= np.float)

    for n in range(len(x_data)):
        temp = (-2)*float(y_data[n]- (b + np.dot(w, x_data[n]) + np.dot(a, x_data[n]**2)))
        b_grad = b_grad + temp
        w_grad = w_grad + temp * x_data[n]
        a_grad = a_grad + temp * x_data[n]**2
    w_grad = w_grad + 2 * lambda_smooth * w   
    a_grad = a_grad + 2 * lambda_smooth * a
    print(str((i+1)*100/iteration) + '%, ' + str(b_grad) )    
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    a_lr = a_lr + a_grad**2    
    b = b - 1/np.sqrt(b_lr) * b_grad
    w = w - 1/np.sqrt(w_lr) * w_grad
    a = a - 1/np.sqrt(a_lr) * a_grad


# Test Data Handling
TestData = pandas.read_csv(sys.argv[2], header=None)

test_input = []
test_output = []


for i in range(0, len(TestData.index), 18):
    temp = TestData.iloc[i+9].loc[2:10].as_matrix()
    
    temp = temp.astype(np.float)  
    test_input.append(temp)



#Test Data
for i in range(len(test_input)):
    predict_elm = int(b + np.dot(w, test_input[i]) + np.dot(a, test_input[i]**2))
    if predict_elm >=0:
        test_output.append(predict_elm)
    else:
        test_output.append(-1)
        
#Output Predict Data
columns = ['id','value']
df = pandas.DataFrame(columns=columns)

for i in range(len(test_output)):
    df.loc[i,'id']='id_'+str(i)
    df.loc[i,'value']=test_output[i]
    
df.to_csv(sys.argv[3], sep=',', index=False)


