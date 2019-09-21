import numpy as np
import pandas as pd

import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def read_db(filename="data/MIMIC_DB_train.csv"):
    data = pd.read_csv(filename, dtype=np.float64)
    # print (data)
    mean_values = data.mean()
    for i, value in enumerate(mean_values):
        float_list = [3,4,5,6,7,8,9]
        if i not in float_list:
            mean_values[i] = round(value)
    # print (mean_values)
    # print (data.describe())
    values = mean_values.to_dict()
    # print (values)
    
    data = data.fillna(value=values)
    # print (data)
    data_array = data.values

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
        self.x_data = torch.from_numpy(xy[:, 1:18]).float()
        self.y_data = torch.from_numpy(xy[:, 18])
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
    means_numpy = np.asarray([0.5, 61.9, 14.3, 4.2, 8.9, 1.0, 0.19, 69.3, 167.2, 0.5, 0.5, 0.5, 0.5, 3, 56.5, 49.5, 45.5])
    stds_numpy = np.asarray([0.5, 9.39, 1.49, 0.41, 0.38, 0.5, 0.49, 10.4, 7.7, 0.5, 0.5, 0.5, 0.5, 3, 7.7, 6.65, 50.3])
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
    train_loader = get_dataloader(is_train=True)

    for i, data in enumerate(train_loader):
        print (i, data)
