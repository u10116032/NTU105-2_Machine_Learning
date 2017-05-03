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
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




#%%
x_val=[]
y_val=[]
train_data = pandas.read_csv(sys.argv[1]) #training data

for i in range(0,4000):
    temp = np.array(list(map(int, train_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_val.append(temp)
    temp = np.zeros((7, ),dtype=np.int)
    temp[int(train_data.loc[i, 'label'])] = 1
    y_val.append(temp)
x_val = np.array(x_val)
y_val = np.array(y_val)

#%%
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
model = load_model('./Final_Model.h5')
#plot_model(model, to_file='model.png')

predict_result = model.predict_classes(x_val, batch_size=200)
conf_mat = confusion_matrix(np.argmax(y_val, axis=1),predict_result)
print(conf_mat)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig('confusion matrix.png')
plt.show()

