#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:14:23 2018

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

# Skicit-learn packages
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing, model_selection, metrics
from sklearn.ensemble import RandomForestRegressor

#XGBoost package
import xgboost as xgb

# quick way of calculating a numeric has for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

# hash a complete column of a pandas dataframe
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

def dataPreprocessing(df, price_log, cat_vars_transf):

    # 1) Numerical Variables
      # Price
       # Logaritmic scale so we reduce the impacte of big differences in Price
    if log_price == True:
        df['price_log'] = np.log(df['price'] + 0.001)
        # Missing Values of Price: First approach will be to input a flag -99
        df['price_log'].fillna(-99, inplace=True)

    # 2) Date-Time Variables
    df['weekday'] = df['activation_date'].dt.weekday
    df['month'] = df.activation_date.dt.month
    df['day'] = df.activation_date.dt.day
    df['week'] = df.activation_date.dt.week
    if del_activation_date == True:
        del df['activation_date']

    # 3) Categorical Variables
    # user_id will be considered as a categorical variable for the time being
    # because user_id cannot be mapped to a numerical variable

    # Array with all of those variables that will be considered as Categorical
    cat_vars = ["region", "city", "parent_category_name",
                "category_name", "user_type", "param_1",
                "param_2", "param_3","user_id"]

    if cat_vars_transf == 'map_to_num':
        for col in cat_vars:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[col].values.astype('str')))
            df[col] = lbl.transform(list(df[col].values.astype('str')))
    if cat_vars_transf == 'hash':
        for col in cat_vars:
            df[col] = df.apply(lambda row: prep_funs.hash_column(row, col),axis=1)

    # Text Variables
      #Title and length of the title
    df['title'] = df['title'].fillna(" ")
    df['title_len'] = df['title'].apply(lambda x : len(x.split()))
     # Description and length of the Description
    df['description'] = df['description'].fillna(" ")
    df['description_len'] = df['description'].apply(lambda x : len(x.split()))

    # Image Variables
     # for the beginning I use only the information if there is an image or not
    df['no_image'] = df['image'].isnull().astype(int)

    return(df)
