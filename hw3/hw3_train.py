# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 00:03:21 2017

@author: halley
"""
#%%
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
#import matplotlib.pyplot as plt
#from scipy import ndimage
#from PIL import Image
#import random
#from sklearn.cross_validation import train_test_split
'''
def gradient_process(img):  
    kernel = np.array([[0,    0,   0],
                       [1,   -2,   1],
                       [0,    0,   0]])            
    gradient_x = ndimage.convolve(img.reshape(48,48), kernel)
    kernel = np.array([[0,    1,   0],
                       [0,   -2,   0],
                       [0,    1,   0]])            
    gradient_y = ndimage.convolve(img.reshape(48,48), kernel)
    gradient = (gradient_x**2+gradient_y**2)**0.5
    gradient = gradient.reshape(1,48*48)
    for index in range(len(gradient)):
        if gradient[0,index]<0:
                gradient[index]=0      
    gradient = gradient.astype(np.int)
    return gradient.reshape(48,48,1)

def sobel_process(img):  
    kernel = np.array([[1,    2,   1],
                       [0,    0,   0],
                       [-1,  -2,  -1]])            
    sobel_x = ndimage.convolve(img.reshape(48,48), kernel)
    kernel = np.array([[1,    0,  -1],
                       [2,    0,  -2],
                       [1,    0,  -1]])            
    sobel_y = ndimage.convolve(img.reshape(48,48), kernel)
    sobel = (sobel_x**2+sobel_y**2)**0.5
    sobel = sobel.reshape(1,48*48)
    for index in range(len(sobel)):
        if sobel[0,index]<0:
                sobel[index]=0      
    sobel = sobel.astype(np.int)
    return sobel.reshape(48,48,1)

def LBP(img):
    img = img.reshape(48,48)
    result = np.zeros((46,46))
    kernel = np.array([[1,    2,   4],
                       [128,  0,   8],
                       [64,  32,  16]]) 
    
    for i in range(1, 46):
        for j in range(1,46):
            temp = np.array((img[i-1:i+2,j-1:j+2] > img[i,j]).astype(np.int))          
            result[i-1,j-1] = sum(sum(temp*kernel))            
    return result.reshape(46,46,1)

def high_pass(img):
    img = img.reshape(48,48)
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])
    highpass_5x5 = ndimage.convolve(img, kernel)
    return highpass_5x5.reshape(48,48,1)

def flip(img):
    img = img.reshape(48,48)
    result = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            result[i,j] = img[i, 47-j]
    return result.reshape(48,48,1) 

def noise(img):
    img = img.reshape(48,48)
    result = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            noise = random.randint(0,5)
            if (result[i,j]+noise) > 255:
                result[i,j] = 255
            else:
                result[i,j] = result[i,j]+noise
    return result.reshape(48,48,1)

def mask(img):
    img = img.reshape(48,48)
    (x,y) = (random.randint(0,42), random.randint(0,42))
    for i in range(x, x+5):
        for j in range(y, y+5):
            img[i,j] = 0
    return img.reshape(48,48,1)
'''    
#%%
train_data = pandas.read_csv(sys.argv[1])
#test_data = pandas.read_csv('test.csv')

x_train = []
y_train = []
x_val=[]
y_val=[]

for i in range(train_data.shape[0]-2000):
    temp = np.array(list(map(int, train_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_train.append(temp)
    temp = np.zeros((7, ),dtype=np.int)
    temp[int(train_data.loc[i, 'label'])] = 1
    y_train.append(temp)


for i in range(train_data.shape[0]-1999,train_data.shape[0]):
    temp = np.array(list(map(int, train_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_val.append(temp)
    temp = np.zeros((7, ),dtype=np.int)
    temp[int(train_data.loc[i, 'label'])] = 1
    y_val.append(temp)

x_train = np.array(x_train)
y_train = np.array(y_train)    
x_val = np.array(x_val)
y_val = np.array(y_val)

#%%
'''
x_test = []
for i in range(test_data.shape[0]):
    temp = np.array(list(map(int, test_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_test.append(temp)
x_test = np.array(x_test)    
'''

#%%

datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True)

datagen.fit(x_train)

model = Sequential()

model.add(Conv2D(50,(3,3),input_shape=(48,48,1)))#42
model.add(Activation('relu'))
model.add(AveragePooling2D((2,2)))#21
model.add(Dropout(0.2))
model.add(Conv2D(75,(3,3)))#15
model.add(Activation('relu'))
model.add(AveragePooling2D((2,2)))#7
model.add(Dropout(0.2))
model.add(Conv2D(150,(3,3)))#15
model.add(Activation('relu'))
model.add(AveragePooling2D((2,2)))#21
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=50,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=200,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=400,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=800,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=1000,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=7,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=400,epochs=200,validation_split=0.1, shuffle=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
                            samples_per_epoch=x_train.shape[0],
                            nb_epoch=400,
                            validation_data=(x_val, y_val))
score = model.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])
    
predict_result = model.predict_classes(x_test, batch_size=200)
predict_result_prob = model.predict_proba(x_test, batch_size=200)

model.save('Final_Model_v3.h5')

#%%
#Output Predict Data
'''
columns = ['id','label']
df = pandas.DataFrame(columns=columns)

for i in range(len(predict_result)):
    df.loc[i,'id']=str(i)
    df.loc[i,'label']=predict_result[i]
df.to_csv('result.csv', sep=',', index=False)
'''
#%%
#print(ndimage.interpolation.rotate(x_train[9],30).shape(0,0))
#plt.imshow(ndimage.interpolation.rotate(x_train[9],30, reshape=False).reshape(48,48), cmap=plt.get_cmap('gray'))
#plt.imshow(flip(x_train[9]).reshape(48,48), cmap=plt.get_cmap('gray'))

#print(y_train[9])
#plt.imshow(x_train[2].reshape(48,48), cmap=plt.get_cmap('gray'))







