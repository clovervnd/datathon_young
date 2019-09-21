import numpy as np
import pandas as pd

import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def read_db(filename="data/MIMIC_DB_train.csv"):
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
            'is_vent': 4}

    data = data.fillna(value = comorb_fill)
    print (data.describe())
    mean_values = data.mean()
    # print (mean_values)



    data = pd.get_dummies(data, columns=["sex", "c_CV","c_IHD", "c_HF", "c_HTN", "c_DM", "c_COPD", "c_CKD", "c_ESRD", "c_HEM", "c_METS", "c_AF", "c_LD", "is_vent" ]) 

    values = mean_values.to_dict()
    # print (values)
    data = data.fillna(value=values)
    # print (data)
    data_array = data.values

    # print (data.describe())
    mean_values = data.mean()
    print (mean_values)

    return data_array


class TestDataset(Dataset):
    """ Test dataset."""

    # Initialize your data, download, etc.
    def __init__(self, filename="data/MIMIC_DB", is_train=True, transform=None):
        if is_train:
            filename = filename + "_train.csv"
        else:
            filename = filename + "_test.csv"
        xy = read_db(filename)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 3:5] + xy[:, 6:]).float()
        self.y_data = torch.from_numpy(xy[:, 5])
        self.y_data[self.y_data > 1] =  1
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
    means_numpy = np.asarray([])
    stds_numpy = np.asarray([])
    # print (x)
    means = torch.from_numpy(means_numpy).float()
    stds = torch.from_numpy(stds_numpy).float()

    transform_x = (x - means) /stds
    return transform_x

def get_dataloader(is_train=True, batch_size=32, shuffle=True, num_workers=1):
    all_data = read_db()
    dataset = TestDataset(is_train = is_train, transform = transform)
    dataloader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    np.set_printoptions(threshold=sys.maxsize)
    read_db();



    # train_loader = get_dataloader(is_train=True)

    # for i, data in enumerate(train_loader):
        # print (i, data)
