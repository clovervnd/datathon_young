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
    if 'HC_label' in whole_csv.columns:
        pos_df = whole_csv[whole_csv.HC_label == 1].sample(frac=1)
        neg_df = whole_csv[whole_csv.HC_label == 0].sample(frac=1)
    
    elif 'death_label' in whole_csv.columns:
        pos_df = whole_csv[whole_csv.death_label == 1].sample(frac=1)
        neg_df = whole_csv[whole_csv.death_label == 0].sample(frac=1)
    
    else:
        raise ValueError("Label Name is not in ['HC_label', 'death_label']")
    
    train_pos = pos_df[int(len(pos_df)*0.3):]
    test_pos = pos_df[:int(len(pos_df)*0.3)]

    train_neg = neg_df[int(len(neg_df)*0.3):]
    test_neg = neg_df[:int(len(neg_df)*0.3)]
     
    train_df = pd.concat([train_pos, train_neg])
    test_df  = pd.concat([test_pos, test_neg])
     
    # save

    print(len(train_df), len(test_df))

    train_file_name = os.path.join(save_path, '%s_DB_train.csv'%phase.upper())
    test_file_name = os.path.join(save_path, '%s_DB_test.csv'%phase.upper())

    print('train file : ', train_file_name)
    print('test file : ', test_file_name)

    train_df.to_csv(train_file_name,index=False); test_df.to_csv(test_file_name,index=False)
    print('saved')

    return

# Example
train_test_split_csv(
    file = './eICU_ver1.5.csv',
    test_ratio=0.3,
    phase='eicu',
    save_path='./'
)
