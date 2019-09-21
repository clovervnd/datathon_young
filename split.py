#!/usr/bin/env python
# coding: utf-8
# Editor : Sang wook Kim, Korea University

import pandas as pd, os, numpy as np

def train_test_split_csv(file, test_ratio, phase, save_path = './'):
    
    print('database type : ', phase)
    print('filename : ', file)
    
    # read csv file
    whole_csv = pd.read_csv(file)
    
    # split
    
    tr_index = int(len(whole_csv) * test_ratio)
    
    test_df, train_df = whole_csv[:tr_index], whole_csv[tr_index:]
    
    # save
    
    print(len(train_df), len(test_df))
    
    train_file_name = os.path.join(save_path, '%s_DB_train.csv'%phase.upper())
    test_file_name = os.path.join(save_path, '%s_DB_test.csv'%phase.upper())
    
    print('train file : ', train_file_name)
    print('test file : ', test_file_name)
    
    train_df.to_csv(train_file_name); test_df.to_csv(test_file_name)
    print('saved')
    
    return 

# Example
train_test_split_csv(
    file = './data/MIMIC_DB_train.csv',
    test_ratio=0.2,
    phase='mimic',
    save_path='../'
)