from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ['dataset', 'net', 'train_loop', 'test_loop']


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
        self.hid3 = nn.Linear(100, 100)
        self.hid4 = nn.Linear(100, 20)
        self.output = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.hid1(x))
        x = self.relu(self.hid2(x))
        x = self.relu(self.hid4(x))
        # x = self.relu(self.hid4(x))
        x = self.sigmoid(self.output(x))
        return x.clamp(min=1e-6, max=1-1e-6)


def train_loop(dataloader, model, optimizer, dict_fp, loss_fn=nn.BCELoss()):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            torch.save(model.state_dict(), dict_fp)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # Use round to convert predictions to class labels
            pred_classes = torch.round(pred)
            correct += (pred_classes == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
