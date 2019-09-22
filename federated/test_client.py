import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import argparse
import sys

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker
from syft.frameworks.torch.federated import utils
import numpy as np
import pandas as pd

import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

LOG_INTERVAL = 25
DATATHON = 1

###########################################
##############  Reference #################
###########################################


def random_oversampling(feature_data, feature_label, random_state):
    X_resampled, y_resampled = \
        RandomOverSampler(random_state = random_state).fit_resample(feature_data, feature_label)

    return X_resampled, y_resampled

def smote(feature_data, feature_label, random_state):
    X_resampled, y_resampled = \
        SMOTE(random_state = random_state).fit_resample(feature_data, feature_label)

    return X_resampled, y_resampled

def read_db(filename="data/MIMIC_DB_train.csv", is_train=True):
    data = pd.read_csv(filename, dtype=np.float64)
    # print (data)
    comorb_fill = {'c_ESRD': 0,
            'c_HF': 0,
            'c_HEM': 0,
            'c_COPD': 0,
            'c_METS': 0,
            'c_LD': 0,
            'c_CKD': 0,
            'c_CV': 0,
            'c_DM': 0,
            'c_AF': 0,
            'c_IHD': 0,
            'c_HTN': 0,
            'is_vent': 4,
            'death_label': 0}

    data = data.fillna(value = comorb_fill)
    # print (data.describe())
    mean_values = data.mean()
    # print (mean_values)



    data = pd.get_dummies(data, columns=["sex", "c_CV","c_IHD", "c_HF", "c_HTN", "c_DM", "c_COPD", "c_CKD", "c_ESRD", "c_HEM", "c_METS", "c_AF", "c_LD", "is_vent" ])

    values = mean_values.to_dict()
    # print (values)
    data = data.fillna(value=values)

    if is_train:
        mean_values = data.mean()
        np_mean = np.asarray(mean_values)
        # print (np_mean.shape)
        np_mean = np.append(np_mean[3:4], np_mean[5:45], axis=0)
        # print (np_mean.shape)
        np.savetxt("train_mean.txt", np_mean, delimiter = ',', fmt = '%f')

        std_values = data.std()
        np_std = np.asarray(std_values)
        # print (np_std.shape)
        np_std = np.append(np_std[3:4], np_std[5:45], axis=0)
        # print (np_std.shape)
        np.savetxt("train_std.txt", np_std, delimiter = ',', fmt = '%f')
    # print (data)
    data_array = data.values

    # print (data.describe())
    mean_values = data.mean()
    std_values = data.std()
    # print (mean_values)
    # print (std_values)

    if is_train:
        print (data_array[:, 4].shape)
        print (Counter(data_array[:, 4]))
        data_array, temp = random_oversampling(data_array, data_array[:, 4].astype(np.int) , np.random.randint(100))

        print (Counter(temp))
        print (data_array.shape)

    return data_array

def transform(x):
    # Normlaize data
    train_mean = np.loadtxt(fname = 'train_mean.txt', delimiter =',')
    train_std = np.loadtxt(fname = 'train_std.txt', delimiter =',')
    means = np.append(train_mean, np.asarray([0.5 for i in range(68)]))
    stds = np.append(train_std, np.asarray([0.5 for i in range(68)]))

    transform_x = (x - means) /stds

    return transform_x


def preprocessed_data(dataset_name, is_train):
    if is_train:
        filename = "data/" + dataset_name + "_train.csv"
    else:
        filename = "data/" + dataset_name + "_test.csv"
    xy = read_db(filename, is_train)
    segment1 =xy[:,3:4]
    segment2 =xy[:, 5:]
    x_data = np.append(segment1, segment2,axis=1)
    y_data = xy[:, 4]

    x_data = transform(x_data)

    return x_data, y_data

print (preprocessed_data("MIMIC_DB", True))



###########################################
###########################################
###########################################
