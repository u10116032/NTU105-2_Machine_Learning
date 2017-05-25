#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:53:24 2017

@author: halley
"""
#%%
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import pickle

test_path = sys.argv[1]
output_path = sys.argv[2]
#%%
#####################
###   parameter   ###
#####################
split_ratio = 0.001
embedding_dim = 100
nb_epoch = 500
batch_size = 64
num_words= 51867


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding='utf-8', errors='ignore') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)



###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################

#%%
### read training and testing data
(_, X_test,_) = read_data(test_path,False)

word_index = pickle.load( open( "word_index.pkl", "rb" ) )

### tokenizer for all data
tokenizer = Tokenizer()
tokenizer.word_index = word_index

#%%
### convert word sequences to index sequence
print ('Convert to index sequences.')
test_sequences = tokenizer.texts_to_sequences(X_test)

### padding to equal length
print ('Padding sequences.')
test_sequences = pad_sequences(test_sequences,maxlen=306)


### build model
print ('Building model.')
model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    #weights=[embedding_matrix],
                    input_length=306,
                    trainable=False))
model.add(GRU(128,activation='tanh',dropout=0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(38,activation='sigmoid'))
model.summary()

adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[f1_score])


model.load_weights('best_weight.hdf5')

tag_list = pickle.load( open( "tag_list.pkl", "rb" ) )
#%%
Y_pred = model.predict(test_sequences)
thresh = 0.4
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    #turn all the prob which is larger than 0.4 to 1 (labeled)
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)







