#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:29:46 2017

@author: halley
"""

###################### Training Part ######################

# xgboost

import pandas as pd
import numpy as np
import xgboost as xgb
import random
import pickle
import sys

status_group = ["functional", "non functional", "functional needs repair"]
date_weight = [365,30,1]

train = pd.read_csv("../pre_train_data.csv")


#################### Feature Engineering ####################

column_labels = ['amount_tsh','funder', 'gps_height', 
'longitude','latitude', 'basin', 'lga', 'population', 
'public_meeting', 'scheme_name','permit','construction_year',
'extraction_type_class','management','management_group','payment', 
'quantity', 'quality_group', 'source','source_type','source_class','waterpoint_type', 'date_recorded']

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

##############################Training part##############################

for model_term in range(5):
    models = []
    models_acc = []
    
    for t in range(10):
        random_num = random.randint(1,100000)
        dtrain = xgb.DMatrix( train[column_labels], label=train["status_group"], missing=0)
        
        print("Asign train to DMatrix: successfully")
          
        clf = xgb.XGBClassifier(n_estimators = 500,
            learning_rate = 0.2,
            objective = 'multi:softmax',
            booster = 'gbtree',
            colsample_bytree = 0.4,
            random_state = random_num)
            
        xgb_params = clf.get_xgb_params()
        xgb_params['num_class'] = 3
        xgb_params['max_depth'] = 12
            
        cv_result = xgb.cv(xgb_params,
        dtrain,
        num_boost_round = 1000,
        nfold = 5,
        metrics = {"merror"},
        maximize = False,
        early_stopping_rounds = 10,
        seed = random_num,
        callbacks = [xgb.callback.print_evaluation(show_stdv = False)]
        )
        
        print("finish cv")
        clf.set_params(max_depth = 14)
        clf.set_params(n_estimators = cv_result.shape[0])
        clf.fit(train[column_labels], train["status_group"], eval_metric= "merror")
            
        # Accuracy
        # accuracy= np.sum(clf.predict(validation[column_labels])==validation["status_group"]) / validation.shape[0]
        # print("Train Accuracy= " , accuracy , "%")
        models.append(clf)
        models_acc.append(cv_result.iloc[cv_result.shape[0]-1]["test-merror-mean"])
        print("finish ", t, "st time.")
       
    # Choose Best Modle
    models_acc = np.array(models_acc)
    best_model = models[np.argmin(models_acc)]

    pickle.dump(best_model, open("xgb_models/model_"+ str(model_term) +".dat", "wb"))
    print("finish writing model", str(model_term))




