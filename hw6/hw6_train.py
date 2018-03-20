# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:04:12 2017

@author: halley
"""
#%%
import math
import pandas 
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, History
from CFModel import CFModel, DeepModel
import numpy as np
from keras import backend as K


def RMSE(y_pred , y_true):
    return K.sqrt(K.mean((y_pred - y_true)**2))



#%%
TRAIN_CSV_FILE = 'train.csv'
TEST_CSV_FILE = 'test.csv'
OUTPUT_CSV_FILE = 'delete.csv'
MODEL_WEIGHTS_FILE = 'delete.h5'
K_FACTORS = 200
RNG_SEED = 1446557
#%%
train_data = pandas.read_csv(TRAIN_CSV_FILE, sep=',')
max_userid = train_data['UserID'].drop_duplicates().max()
max_movieid = train_data['MovieID'].drop_duplicates().max()
print(len(train_data), 'ratings loaded.')

#%%

shuffled_ratings = train_data.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['UserID'].values
print ('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['MovieID'].values
print ('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['Rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)

#%% 
#normalization Rating

Ratings_mean = np.mean(Ratings)
Ratings_dev = np.std(Ratings)

#Ratings = (Ratings - Ratings_mean)/Ratings_dev

#%%
model = CFModel(max_userid+1, max_movieid+1, K_FACTORS)
callbacks = [EarlyStopping('val_RMSE', patience= 10, verbose=1, mode='min'), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, 
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_RMSE',
                             mode='min')]
model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
history = model.fit([Users, Movies,Users, Movies], Ratings, epochs= 5000, batch_size= 1000 ,validation_split=.1, verbose=1, callbacks=callbacks)


#%%
'''
test_data = pandas.read_csv(TEST_CSV_FILE, sep=',')
test_userid = test_data['UserID']
test_movieid = test_data['MovieID']

columns = ['TestDataID','Rating']
df = pandas.DataFrame(columns=columns)

trained_model = CFModel(max_userid+1, max_movieid+1, K_FACTORS)
trained_model.load_weights(MODEL_WEIGHTS_FILE)

result = trained_model.predict([np.array(test_userid),np.array(test_movieid), np.array(test_userid),np.array(test_movieid)]).flatten()

for i in range(len(test_userid)):   
    temp = result[i] #* Ratings_dev + Ratings_mean
    if temp < 0:
        temp = 0
    if temp > 5:
        temp = 5
    df.loc[i,'TestDataID'] = str(i+1)
    df.loc[i,'Rating'] = temp
    print("test data[",i,"]: ",temp)
#%%
df.to_csv(OUTPUT_CSV_FILE, sep=',', index=False)
print('min val_RMSE: ', np.min(history.history['val_RMSE']))
'''
#%%

x = np.arange(0,len(history.history['val_RMSE']))
plt.plot(x,history.history['RMSE'], label='train RMSE')
plt.plot(x,history.history['val_RMSE'], label='valid_RMSE')
plt.ylabel("RMSE val")
plt.xlabel("# of echo")
plt.title("Training Procedure")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('history_MF.png')

plt.show()










    
