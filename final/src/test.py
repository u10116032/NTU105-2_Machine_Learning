#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:48:50 2017

@author: halley
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys


status_group = ["functional", "non functional", "functional needs repair"]
date_weight = [365,30,1]

#%%
##############################feature processing part##############################
train = pd.read_csv("../pre_train_data.csv")
test = pd.read_csv("../pre_test.csv")
test = test.fillna(test.median())

column_labels = ['amount_tsh','funder', 'gps_height', 
'longitude','latitude', 'basin', 'lga', 'population', 
'public_meeting', 'scheme_name','permit','construction_year',
'extraction_type_class','management','management_group','payment', 
'quantity', 'quality_group', 'source','source_type','source_class','waterpoint_type', 'date_recorded']


#%%
########################## test preprocessing ####################################

def date_preprocess(dataframe, stat='training'):
    dataframe_date = np.asarray(dataframe["date_recorded"], dtype=str)
    dataframe_date = np.array(np.core.defchararray.split(dataframe_date, sep='-'))
    dataframe_date = list(dataframe_date)
    dataframe_date = np.array(dataframe_date, dtype=int)
    dataframe_date = dataframe_date * date_weight
    dataframe_date = np.sum(dataframe_date, axis=1)
    dataframe['date_recorded'] = dataframe_date
    min_date = np.min(np.array(dataframe["date_recorded"]))
    if stat == 'training':
        dataframe["date_recorded"] = (dataframe["date_recorded"] - min_date).astype(int)
    return dataframe, min_date

########## date_recorded for trainig data ##########

train, min_date = date_preprocess(train,stat='training')
print("Features date_recorded for training: successfully")

########## date_recorded for testing data ##########

test,_ = date_preprocess(test,stat='testing')
test["date_recorded"] = (test["date_recorded"]- min_date).astype(int)
print("Features date_recorded for testing: successfully")

#%%
###################### Ensemble ###############################

model_xgb_1 = pickle.load(open("xgb_models/model_0.dat", "rb"))
model_xgb_2 = pickle.load(open("xgb_models/model_1.dat", "rb"))
model_xgb_3 = pickle.load(open("xgb_models/model_2.dat", "rb"))
model_xgb_4 = pickle.load(open("xgb_models/model_3.dat", "rb"))
model_xgb_5 = pickle.load(open("xgb_models/model_4.dat", "rb"))
model_xgb_6 = pickle.load(open("xgb_models/model_840.dat", "rb"))

result_1 = model_xgb_1.predict(test[column_labels])
result_2 = model_xgb_2.predict(test[column_labels])
result_3 = model_xgb_3.predict(test[column_labels])
result_4 = model_xgb_4.predict(test[column_labels])
result_5 = model_xgb_5.predict(test[column_labels])
result_6 = model_xgb_6.predict(test[column_labels])

#with random forest
import gc
import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

column_labels_rf = list(test.columns.values)
column_labels_rf.remove("id")
column_labels_rf.remove("date_recorded")
#column_labels_rf.remove("status_group")

clf_1 = joblib.load('rf_models/clf_1.pkl','r')
clf_2 = joblib.load('rf_models/clf_2.pkl','r')
clf_3 = joblib.load('rf_models/clf_3.pkl','r')
clf_4 = joblib.load('rf_models/clf_4.pkl','r')
clf_5 = joblib.load('rf_models/clf_5.pkl','r')

prediction_1 = clf_1.predict(test[column_labels_rf])
prediction_2 = clf_2.predict(test[column_labels_rf])
prediction_3 = clf_3.predict(test[column_labels_rf])
prediction_4 = clf_4.predict(test[column_labels_rf])
prediction_5 = clf_5.predict(test[column_labels_rf])


############# Hard Label Majority Vote ############


hard_label_final = np.zeros((test.shape[0],3), dtype= int)
for i in range(test.shape[0]):
    hard_label_final[i, result_1[i]] += 1
    hard_label_final[i, result_2[i]] += 1
    hard_label_final[i, result_3[i]] += 1
    hard_label_final[i, result_4[i]] += 1
    hard_label_final[i, result_5[i]] += 1
    hard_label_final[i, result_6[i]] += 1
    hard_label_final[i, prediction_1[i]] += 1
    hard_label_final[i, prediction_2[i]] += 1
    hard_label_final[i, prediction_3[i]] += 1
    hard_label_final[i, prediction_4[i]] += 1
    hard_label_final[i, prediction_5[i]] += 1

xgb_rf_hard_ensemble_result = []
for i in range(hard_label_final.shape[0]):
    xgb_rf_hard_ensemble_result.append(np.argmax(hard_label_final[i]))
    
#%%
############# Making submission file #############

# Dataframe as per submission format
submission = pd.DataFrame({
			"id": test["id"],
			"status_group": xgb_rf_hard_ensemble_result
		})
for i in range(len(status_group)):
	submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
print("Dataframe as per submission format: successfully")

# Store submission dataframe into file
submission.to_csv(sys.argv[1], index = False)
print("Store submission dataframe into file: successfully")



