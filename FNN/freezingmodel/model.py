from torch.utils.data import Dataset
import torch
import torch.nn as nn

__all__ = ['dataset', 'net']


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class net(nn.Module):
    def __init__(self, input_size):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.hid1 = nn.Linear(100, 100)
        self.hid2 = nn.Linear(100, 100)
        self.hid3 = nn.Linear(100, 20)
        self.output = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.hid1(x)
        x = self.relu(x)
        x = self.hid2(x)
        x = self.relu(x)
        x = self.hid3(x)
        x = self.relu(x)
        x = self.sigmoid(self.output(x))
        return x
