#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:39:05 2017

@author: halley
"""

import gc
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

###Training part###
# Traning data
train = pd.read_csv("../pre_train_data.csv")
print("Traning data: successfully")

# Features for trainig
column_labels = list(train.columns.values)
column_labels.remove("id")
column_labels.remove("date_recorded")
column_labels.remove("status_group")

status_group = ["functional", "non functional", "functional needs repair"]
print("Features for trainig: successfully")

#%%
# Assign data for validation
amount = int(0.8*len(train))
validation = train[amount:]
#train = train[:amount]
print("Assign data for validation: successfully")

# Classifier
# clf = tree.DecisionTreeClassifier()
clf_1 = RandomForestClassifier(n_estimators = 1200, warm_start=True, criterion='entropy', max_features='sqrt', min_samples_split= 2, min_samples_leaf= 2, random_state=123)
clf_2 = RandomForestClassifier(n_estimators = 1200, warm_start=True, criterion='entropy', max_features='sqrt', min_samples_split= 2, min_samples_leaf= 2, random_state=234)
clf_3 = RandomForestClassifier(n_estimators = 1200, warm_start=True, criterion='entropy', max_features='sqrt', min_samples_split= 2, min_samples_leaf= 2, random_state=345)
clf_4 = RandomForestClassifier(n_estimators = 1200, warm_start=True, criterion='entropy', max_features='sqrt', min_samples_split= 2, min_samples_leaf= 2, random_state=456)
clf_5 = RandomForestClassifier(n_estimators = 1200, warm_start=True, criterion='entropy', max_features='sqrt', min_samples_split= 2, min_samples_leaf= 2, random_state=567)

print("Classifier: successfully")
#%%

# Traning
clf_1.fit(train[column_labels], train["status_group"])
clf_2.fit(train[column_labels], train["status_group"])
clf_3.fit(train[column_labels], train["status_group"])
clf_4.fit(train[column_labels], train["status_group"])
clf_5.fit(train[column_labels], train["status_group"])


print("Traning: successfully")

#%%
# Accuracy
accuracy=[]
accuracy.append(accuracy_score(clf_1.predict(validation[column_labels])
, validation["status_group"]))
accuracy.append(accuracy_score(clf_2.predict(validation[column_labels])
, validation["status_group"]))
accuracy.append(accuracy_score(clf_3.predict(validation[column_labels])
, validation["status_group"]))
accuracy.append(accuracy_score(clf_4.predict(validation[column_labels])
, validation["status_group"]))
accuracy.append(accuracy_score(clf_5.predict(validation[column_labels])
, validation["status_group"]))
print("Accuracy = " , accuracy)
print("Accuracy: successfully")


#%%
from sklearn.externals import joblib
joblib.dump(clf_1, './rf_models/clf_1.pkl') 
joblib.dump(clf_2, './rf_models/clf_2.pkl') 
joblib.dump(clf_3, './rf_models/clf_3.pkl') 
joblib.dump(clf_4, './rf_models/clf_4.pkl') 
joblib.dump(clf_5, './rf_models/clf_5.pkl') 
