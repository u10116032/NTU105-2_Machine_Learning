#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:06:01 2017

@author: halley
"""
#%%
from keras import backend as K
import numpy as np
from CFModel import DeepModel, CFModel
from sklearn.manifold import TSNE
import pandas

MAX_MOVIEID = 3952
MAX_USERID = 6040
K_FACTORS = 200

model = CFModel(MAX_USERID+1, MAX_MOVIEID+1, K_FACTORS)
model.load_weights('./MF_Model/model_MF.h5')

#%%
train_data = pandas.read_csv('train.csv', sep=',')
train_userid = train_data['UserID']
train_movieid = train_data['MovieID']

#%%
inp = model.input #input place holders
output = model.get_layer(name='movie_embedded').output #get first layer
functor = K.function(inp + [ K.learning_phase()], [output]) 

print('start dealing with embedding layer(movieId) output')
movie_embedded_outputs = []
for i in range(5000):
    temp = functor([ np.array([train_userid[i]]).reshape(1,1), np.array([train_movieid[i]]).reshape(1,1), np.array([train_userid[i]]).reshape(1,1), np.array([train_movieid[i]]).reshape(1,1), 0 ])
    movie_embedded_outputs.append(temp)
print('finish dealing output layer, start TSNE')    
movie_embedded_outputs = np.array(movie_embedded_outputs).reshape(5000, K_FACTORS) 
#%%
model = TSNE(n_components=2)
np.set_printoptions(suppress=True)
movie_tsne = model.fit_transform(movie_embedded_outputs) 
print('finish TSNE, start classifying movie tag to y_label')
#%%
y_label = []
movie_data = pandas.read_csv('movies.csv', sep='::')
movie_tags = np.array(movie_data['Genres'], dtype= np.str)
movie_id = np.array(movie_data['movieID'], dtype= np.int)
#%%
for i in range(5000):
    idx = np.where(movie_id==train_movieid[i])[0][0]
    if (movie_tags[idx].find('Thriller')!=-1) or (movie_tags[idx].find('Horror')!=-1) or (movie_tags[idx].find('Crime')!=-1 ):
        y_label.append('red')
    elif (movie_tags[idx].find('Drama')!=-1) or (movie_tags[idx].find('Musical')!=-1):
        y_label.append('green')
    else:
        y_label.append('blue')
#%% plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.scatter(movie_tsne[:,0], movie_tsne[:,1], c=y_label, alpha=.5)
patch_0 = mpatches.Patch(color='red', label='Thriller, Horror, Crime')
patch_1 = mpatches.Patch(color='green', label='Drama, Musical')
patch_2 = mpatches.Patch(color='blue', label='Others')
plt.legend(handles=[patch_0, patch_1, patch_2], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True)
plt.title('5000 train data distribution')
plt.show()

    
