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

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

LOG_INTERVAL = 25
DATATHON = 0

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


class TestDataset(Dataset):
    """ Test dataset."""

    # Initialize your data, download, etc.
    def __init__(self, filename="data/MIMIC_DB", is_train=True, transform=None):
        if is_train:
            filename = filename + "_train.csv"
        else:
            filename = filename + "_test.csv"
        xy = read_db(filename, is_train)
        self.len = xy.shape[0]

        segment1 =xy[:,3:4] 
        segment2 =xy[:, 5:]
        # print (segment1.shape)
        # print (segment2.shape)
        temp_x_data = np.append(segment1, segment2,axis=1)
        # print (temp_x_data.shape)
        self.x_data = torch.from_numpy(temp_x_data).float()
        self.y_data = torch.from_numpy(xy[:, 4])
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
    means_numpy = np.append(train_mean, np.asarray([0.5 for i in range(68)]))
    stds_numpy = np.append(train_std, np.asarray([0.5 for i in range(68)]))
    # print (x)
    means = torch.from_numpy(means_numpy).float()
    stds = torch.from_numpy(stds_numpy).float()

    transform_x = (x - means) /stds

    return transform_x

def get_dataloader(is_train=True, batch_size=32, shuffle=True, num_workers=1):
    # all_data = read_db()
    dataset = TestDataset(is_train = is_train, transform = transform)
    # dataset = TestDataset(is_train = is_train)
    dataloader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    return dataloader



###########################################
###########################################
###########################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if DATATHON :
            super(Net, self).__init__()
            self.fc1 = nn.Linear(109, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 200)
            self.fc4 = nn.Linear(200, 200)
            self.fc5 = nn.Linear(200, 200)
            self.fc6 = nn.Linear(200, 100)
            self.fc7 = nn.Linear(100, 10)
            self.fc_final = nn.Linear(10, n_class)
        else:
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        
    def forward(self, x):
        if DATATHON:
            in_size = x.size(0)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc3(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc5(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc6(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc7(x))
            x = F.dropout(x, training=self.training)
            x = self.fc_final(x)
            return F.log_softmax(x)
        else:
            x = f.relu(self.conv1(x))
            x = f.max_pool2d(x, 2, 2)
            x = f.relu(self.conv2(x))
            x = f.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = f.relu(self.fc1(x))
            x = self.fc2(x)
            return f.log_softmax(x, dim=1)

    def softmax(self, x):
        in_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc5(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_final(x)
        return F.softmax(x)



def train_on_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches

    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps

    Returns:
        model, loss: obtained model and loss after training

    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker

    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve

    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]

    """
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches


def train(model, device, federated_train_loader, lr, federate_after_n_batches):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        logger.debug(
            "Starting training round, batches [{}, {}]".format(counter, counter + nr_batches)
        )
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, lr
                )
            else:
                data_for_all_workers = False
        counter += nr_batches
        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
    return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,
        help="number of training steps performed on each remote worker " "before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will " "be started in verbose mode",
    )
    parser.add_argument(
        "--use_virtual", action="store_true", help="if set, virtual workers will be used"
    )

    args = parser.parse_args(args=args)
    return args


def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

    workers = [alice, bob, charlie]

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if DATATHON:
        federated_train_loader = sy.FederatedDataLoader(
                TestDataset(is_train=is_train, transform=transform
                    ).federate(tuple(workers)),
            batch_size=args.batch_size,
            shuffle=True,
            iter_per_worker=True,
            **kwargs,
                )

        test_loader = get_dataloader(is_train=False, batch_size=batch_size)

    else:
        federated_train_loader = sy.FederatedDataLoader(
            datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ).federate(tuple(workers)),
            batch_size=args.batch_size,
            shuffle=True,
            iter_per_worker=True,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs,
        )

    model = Net().to(device)

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
