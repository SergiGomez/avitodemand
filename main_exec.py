#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 18:11:07 2018

@author: sergigomezpalleja
"""

# General packages
import pandas as pd
import numpy as np
import os
import gc
import time
import random
import importlib

# Sk-learn packages
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing, model_selection, metrics
from sklearn.ensemble import RandomForestRegressor

#XGBoost
import xgboost as xgb

# Import our own functions
import kaggle_preprocessing as prep_funs
import models
#importlib.reload(prep_funs)

time_ini_exec = time.time()

print(" %s All packages loaded" % (time.ctime()))

print("Current Working Directory: ", os.getcwd())

log_price = True
val_set_type = 'random_subset'
del_activation_date = True
cat_vars_transf = 'hash'
model_name = 'xgb'

print("\t log_price :",log_price)
print("\t val_set_type :",val_set_type)
print("\t del_activation_date :",del_activation_date)
print("\t cat_vars_transf :",cat_vars_transf)
print("\t model_name :",model_name)

time_read_1 = time.time()
train = pd.read_csv("train.csv", parse_dates=['activation_date'])
test = pd.read_csv("test.csv", parse_dates = ['activation_date'])
time_read_2 = time.time()
print("Time to read Datasets: %s sec" % (round(time_read_2 - time_read_1,2)))

# I keep the item id of the test
id_test = test['item_id']
id_test = pd.DataFrame(id_test, columns = ['item_id'])

train['set'] = 'train'
test['set'] = 'test'
test['deal_probability']= 0
df = pd.concat([train,test],axis=0)
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

# Split 1: Train / Validation / Test --> DF
if val_set_type == 'time_stamp':
    cond_val_1 = df['activation_date'] >= pd.to_datetime('2017-04-08')
    cond_val_2 = df['set'] != 'test'
    df.loc[cond_val_1 & cond_val_2,'set'] = 'val'
elif val_set_type == 'random_subset':
    val_df = df.loc[df['set'] == 'train'].sample(frac=0.1)
    val_df['set'] = 'val'
    id_val = val_df['item_id'].values
    df.loc[df['item_id'].isin(id_val), 'set'] = 'val'
    del val_df

del train
del test
gc.collect()

print("Percentage of Train / Dev / Test : ",
     100*df.loc[df['set'] == 'train'].shape[0] / df.shape[0],
     100*df.loc[df['set'] == 'val'].shape[0] / df.shape[0],
     100*df.loc[df['set'] == 'test'].shape[0] / df.shape[0])

# Numerical Variables
  # Price
   # Log Scale: We reduce the impacte of big differences in Price
if log_price == True:
    df['price_log'] = np.log(df['price'] + 0.001)
    # Missing Values of Price: First approach will be to input a flag -99
    df['price_log'].fillna(-99, inplace=True)
   #Item_seq_number

#Date-Time Variables
df['weekday'] = df['activation_date'].dt.weekday
df['month'] = df.activation_date.dt.month
df['day'] = df.activation_date.dt.day
df['week'] = df.activation_date.dt.week
if del_activation_date == True:
    del df['activation_date']

#Categorical Variables

# I WILL CONSIDER USER_ID AS A CATEGORICAL VARIABLE FOR THE TIME BEING !!
# because user_id cannot be mapped to a numerical variable

cat_vars = ["region", "city", "parent_category_name",
            "category_name", "user_type", "param_1",
            "param_2", "param_3","user_id"]

if cat_vars_transf == 'map_to_num':
    for col in cat_vars:
        #print("col name is: ",col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[col].values.astype('str')))
        df[col] = lbl.transform(list(df[col].values.astype('str')))
if cat_vars_transf == 'hash':
    for col in cat_vars:
        df[col] = df.apply(lambda row: prep_funs.hash_column(row, col),axis=1)

# Text Variables
  #Title
    #Length of the Title
df['title'] = df['title'].fillna(" ")
df['title_len'] = df['title'].apply(lambda x : len(x.split()))
 # Description
   # Length of the Description
df['description'] = df['description'].fillna(" ")
df['description_len'] = df['description'].apply(lambda x : len(x.split()))

# Image Variables
   #Image
# for the beginning I use only the information if there is an image or not
df['no_image'] = df['image'].isnull().astype(int)
   #Image_top1

#Variables that we remove for a first Model
cols_to_drop = ["item_id","title", "description", "image","price","image_top_1"]
df_Model = df.drop(cols_to_drop, axis=1)

# Split 2: DF --> Train / Validation / Test
train_Model = df_Model.loc[df_Model['set'] == 'train']
val_Model = df_Model.loc[df_Model['set'] == 'val']
test_Model = df_Model.loc[df_Model['set'] == 'test']
# I remove the deal_prob of the test !
del test_Model['deal_probability']

print("shape of train: ",train_Model.shape)
print("shape of val: ",val_Model.shape)
print("shape of test: ",test_Model.shape)

del train_Model['set']
del test_Model['set']
del val_Model['set']

# Split Train and Val with X and Y
train_X = train_Model.copy()
val_X = val_Model.copy()
del train_X['deal_probability']
del val_X['deal_probability']
train_Y = train_Model['deal_probability'].values
val_Y = val_Model['deal_probability'].values
test_X = test_Model.copy()

###

target_var = 'deal_probability'
#IDcol = 'ID'

dataset = {'train': train_Model,
           'val': val_Model,
           'test': test_Model}

params =  {'objective': 'reg:linear',
           'n_estimators': 1,
           'learning_rate': 0.05,
           'gamma': 0,
           'subsample': 0.75,
           'colsample_bytree': 1,
           'max_depth': 7,
           'n_jobs': -1,
           'silent': False}

###

#Modelling Stage
  # Features of our model

  # Fitting the Model

time_modelstart = time.time()

if model_name == 'rf':
    print("Simple Random Forest")
    # Instantiate model with 1000 decision trees and max_depth = 5
    rf = RandomForestRegressor(n_estimators = 100, max_depth = 5,
                               random_state = 123, n_jobs = -1)
    # Train the model on training data
    rf.fit(train_X, train_Y)
    # Use the forest's predict method on the test data
    pred_val = rf.predict(val_X)
    print("number of observations with pred_val < 0: ",len(np.where(pred_val < 0)[0]))
    print("number of observations with pred_val > 1: ",len(np.where(pred_val > 1)[0]))
    rmse = np.sqrt(sk.metrics.mean_squared_error(val_Y, pred_val))
    print("The RMSE of the Val is: ", rmse)
    # Prediction of the Test set
    test_pred = rf.predict(test_Model)
elif model_name == 'lgb':
    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 15,
        # 'num_leaves': 31,
        # 'feature_fraction': 0.65,
        'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'learning_rate': 0.02,
        'verbose': 1
        }
    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(train_X, train_Y,
                feature_name=tfvocab,
                categorical_feature = categorical)
    lgval = lgb.Dataset(val_X, val_Y,
                feature_name=tfvocab,
                categorical_feature = categorical)

    lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=1500,
            valid_sets=[lgtrain, lgval],
            valid_names=['train','val'],
            early_stopping_rounds=100,
            verbose_eval=100
            )
    print('RMSE:', np.sqrt(metrics.mean_squared_error(val_Y, lgb_clf.predict(val_X))))

elif model_name == 'xgb':

    model_trained = models.create_model(dataset, model_name, target_var, params)
    pred_val = model_trained.predict(val_X)
    print("number of observations with pred_val < 0: ",len(np.where(pred_val < 0)[0]))
    print("number of observations with pred_val > 1: ",len(np.where(pred_val > 1)[0]))
    rmse = np.sqrt(sk.metrics.mean_squared_error(val_Y, pred_val))
    print("The RMSE of the Val is: ", rmse)
    # Prediction of the Test set
    test_pred = model_trained.predict(test_X)

print("Model Runtime: %0.2f Minutes"%((time.time() - time_modelstart)/60))

test_pred_df = pd.DataFrame(test_pred, columns = ['deal_probability'])
test_sub = pd.concat([id_test,test_pred_df], axis = 1)
sub_file = pd.read_csv('sample_submission.csv')
del sub_file['deal_probability']
sub_file = sub_file.merge(test_sub, left_on = 'item_id', right_on = 'item_id', how = 'left')
sub_file['deal_probability'].clip(0.0, 1.0, inplace=True)
sub_file.to_csv('sub_file_01.csv', index=False)

print("------ Execution Done -------")
print("Total Execution Tiem: %0.2f Minutes"%((time.time() - time_ini_exec)/60))
