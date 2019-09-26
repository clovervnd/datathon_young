import sys

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

outcome_index =0 # 0 is death label, 1 is HC 
is_mimic = 1

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
    comorb_fill = {'c_HF': 0,
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
            'death_label': 0,
            'HC_label':0}

    data = data.fillna(value = comorb_fill)
    # print (data.describe())
    mean_values = data.mean()



    data = pd.get_dummies(data, columns=["sex", "c_CV","c_IHD", "c_HF", "c_HTN", "c_DM", "c_COPD", "c_CKD", "c_HEM", "c_METS", "c_AF", "c_LD", "is_vent" ]) 

    values = mean_values.to_dict()
    # print (values)
    data = data.fillna(value=values)
    print (data.mean())

    if is_train:
        mean_values = data.mean()
        np_mean = np.asarray(mean_values)
        np_mean = np_mean[5:47]
        # print (np_mean.shape)
        np.savetxt("train_mean.txt", np_mean, delimiter = ',', fmt = '%f')

        std_values = data.std()
        np_std = np.asarray(std_values)
        np_std = np_std[5:47]
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
        # print (data_array[:, outcome_index].shape)
        print (Counter(data_array[:, outcome_index]))
        data_array, temp = random_oversampling(data_array, data_array[:, outcome_index].astype(np.int) , np.random.randint(100))
        
        print (Counter(temp))
        # print (data_array.shape)

    return data_array


class TestDataset(Dataset):
    """ Test dataset."""

    # Initialize your data, download, etc.
    def __init__(self, filename="data/EICU_DB", is_train=True, transform=None):
        if is_train:
            filename = filename + "_train.csv"
        else:
            filename = filename + "_test.csv"
        xy = read_db(filename, is_train)
        self.len = xy.shape[0]

        # segment1 =xy[:,3:4] 
        # segment2 =xy[:, 5:]
        # print (segment1.shape)
        # print (segment2.shape)
        # temp_x_data = np.append(segment1, segment2,axis=1)
        # print (temp_x_data.shape)
        self.x_data = torch.from_numpy(xy[:,5:]).float()
        self.y_data = torch.from_numpy(xy[:, outcome_index])
        self.transform = transform

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]

        if self.transform :
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len


def transform(x):
    # Normlaize data
    train_mean = np.loadtxt(fname = 'train_mean.txt', delimiter =',')
    train_std = np.loadtxt(fname = 'train_std.txt', delimiter =',')
    if is_mimic == 1:
        means_numpy = np.append(train_mean, np.asarray([0.5 for i in range(69)]))
        stds_numpy = np.append(train_std, np.asarray([0.5 for i in range(69)]))
    else:
        means_numpy = np.append(train_mean, np.asarray([0.5 for i in range(63)]))
        stds_numpy = np.append(train_std, np.asarray([0.5 for i in range(63)]))
    # print (x)
    means = torch.from_numpy(means_numpy).float()
    stds = torch.from_numpy(stds_numpy).float()

    transform_x = (x - means) /stds

    return transform_x

def get_dataloader(is_train=True, batch_size=32, shuffle=True, num_workers=1, is_mimic=1):
    # all_data = read_db()
    if is_mimic ==1 : 
        dataset = TestDataset(filename = "data/MIMIC_DB", is_train = is_train, transform = transform)
    else:
        dataset = TestDataset(filename = "data/EICU_DB", is_train = is_train, transform = transform)

    # dataset = TestDataset(is_train = is_train)
    dataloader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    return dataloader


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)
    # read_db();



    train_loader = get_dataloader(is_train=True)

    for i, (data,target) in enumerate(train_loader):
        if i ==1:
            print (i)
            print ("data: ", data)
            print ("target: ", target)
