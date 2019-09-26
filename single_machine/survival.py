from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics as sk

import dataloader_improved as dl
import numpy as np
from dt_visualize import save_confusion_matrix, save_roc_curve
# Training settings
batch_size = 64
n_class=2
is_mimic = 1


train_loader = dl.get_dataloader(is_train=True, batch_size=batch_size, is_mimic=0)
test_loader = dl.get_dataloader(is_train=False, batch_size=batch_size, is_mimic=1)


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        if is_mimic == 1:
            self.fc1 = nn.Linear(111, 200)
        else:
            self.fc1 = nn.Linear(105, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.fc_final = nn.Linear(10, n_class)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_final(x)

        return F.log_softmax(x)
    def softmax(self, x):
        in_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_final(x)

        return F.softmax(x)


model = Net()

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target.long())
        # print (data, target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    total_softmax = np.asarray([])
    total_target = np.asarray([])
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target.long())
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        softmax = model.softmax(data)
        total_softmax = np.append (total_softmax, softmax.data.numpy())
        total_target = np.append (total_target, target.numpy())

    test_loss /= len(test_loader.dataset)
    total_softmax = np.reshape(total_softmax,(-1,2))
    print (total_softmax)
    print (total_target)

    np.savetxt(str(is_mimic) + "_total_softmax.txt", total_softmax, delimiter = ',', fmt = '%f')
    np.savetxt(str(is_mimic) + "_total_target.txt", total_target, delimiter = ',', fmt = '%f')
    # Draw AUROC curve
    save_roc_curve(total_target, total_softmax)
    # Draw Confusion Matrix
    save_confusion_matrix(total_target, total_softmax)

    auroc = sk.roc_auc_score(total_target, total_softmax[:,1])
    print ("AUROC: ", auroc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# test()

for epoch in range(1, 5):
    train(epoch)
    test()
