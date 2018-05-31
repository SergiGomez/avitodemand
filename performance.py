#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:56:12 2018

@author: sergigomezpalleja
"""

#perf_df = pd.DataFrame([[0.2421161998393251, 0.2475, 'simple_rf_ntrees100_depth5',
#                         2.05]],
#                      columns = ['val_error','test_error','model_desc', 'time(min)'])


perf_df = pd.read_csv('track_performance.csv', index_col=0)

new_iter = pd.DataFrame([[0.24137015012234506, 0.2466, 'simple_rf_ntrees100_depth5_hash',
                         8.89]],
                      columns = ['val_error','test_error','model_desc', 'time(min)'])

perf_df = pd.concat([perf_df,new_iter], axis = 0)

perf_df.to_csv('track_performance.csv', index=True)