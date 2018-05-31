#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:14:23 2018

@author: sergigomezpalleja
"""
import random

# quick way of calculating a numeric has for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

# hash a complete column of a pandas dataframe    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')
