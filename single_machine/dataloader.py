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
    std_values = data.std()
    # print (mean_values)
    print (std_values)

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
    means_numpy = np.asarray([57.659519
   ,             22.437066
   ,             24.726187
   ,             2.316912
   ,             2.594240
   ,             1.171477
   ,             1.400061
   ,             32.893292
   ,             38.181856
   ,             11.181713
   ,             12.794087
   ,             1.787762
   ,             3.019037
   ,           215.048044
   ,           251.647426
   ,             3.829635
   ,             4.684962
   ,             14.396185
   ,             16.178659
   ,           136.832387
   ,           140.202057
   ,             21.579646
   ,             25.661354
   ,             10.877764
   ,             13.804529
   ,           1478.901961
   ,           104.141938
   ,             93.356222
   ,             46.325566
   ,             78.334767
   ,             26.982716
   ,             12.200633
   ,             36.130542
   ,             37.429889
   ,             92.021236
   ,           109.439431
   ,           273.858638
   ,             54.364061
   ,             7.314209
   ,           242.081106
   ,             48.110817
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,   0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5
    ,    0.5])
    stds_numpy = np.asarray([     14.835527
,            3.992239
,            3.595994
,            2.479988
,            2.732242
,            1.044750
,            1.301933
,            8.399403
,            6.847880
,            2.809906
,            2.339725
,            0.781326
,            1.617444
,            97.454224
,          104.897316
,            0.577061
,            0.826919
,            3.181122
,            6.396289
,            4.252793
,            3.996271
,            15.572532
,            17.864576
,            5.628392
,            7.708330
,            47.682721
,            11.129786
,            9.251199
,            6.331849
,            5.936066
,            3.351123
,            1.930847
,            0.403784
,            0.429330
,            3.510093
,            19.100130
,          5772.594907
,            13.466272
,            0.084561
,          112.088525
,            9.838691
,        0.5
,        0.5
,        0.5
,        0.5
,        0.5
,        0.5
,        0.5
,        0.5
,        0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5
,    0.5])
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
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=sys.maxsize)
    read_db();



    train_loader = get_dataloader(is_train=True)

    for i, data in enumerate(train_loader):
        print (i, data)
