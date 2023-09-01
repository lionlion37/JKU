import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import os


torch.manual_seed(seed=73)
np.random.seed(seed=73)


class nn_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(169, 500)
        self.nonlin1 = nn.Sigmoid()
        self.layer2 = nn.Linear(500, 4)
        self.output = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.nonlin1(x)
        x = self.layer2(x)
        output = self.output(x)
        return output


def train(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.modules.loss, device: torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in data_loader:
        input, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print(model(input), target)

    out_loss = running_loss / len(data_loader)
    accu = 100. * correct / total

    return out_loss, accu


def evaluate(model: nn.Module, data_loader: DataLoader, loss_function: nn.modules.loss, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in data_loader:
        input, target = data[0].to(device), data[1].to(device)
        output = model(input)
        loss = loss_function(output, target)

        running_loss += loss
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    out_loss = running_loss/len(data_loader)
    accu = 100. * correct/total

    return out_loss, accu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = pd.read_pickle(os.path.join('data', 'X_train.pkl'))
X_test = pd.read_pickle(os.path.join('data', 'X_test.pkl'))
y_train = pd.read_pickle(os.path.join('data', 'y_train.pkl'))
y_test = pd.read_pickle(os.path.join('data', 'y_test.pkl'))


X_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train -= 1
y_tensor = torch.tensor(y_train.values).type(torch.LongTensor)

train_data = TensorDataset(X_tensor, y_tensor)

X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test -= 1
y_test_tensor = torch.tensor(y_test.values).type(torch.LongTensor)

test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=10,
                          num_workers=0)

test_loader = DataLoader(test_data,
                          shuffle=True,
                          batch_size=10,
                          num_workers=0)

model = nn_classifier()

loss_function = nn.CrossEntropyLoss(reduction='mean')
lr = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


epochs = 50
for update in range(epochs):
    print(f'Epoch:{update+1}')
    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_function, device)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    print(f'Training loss: {train_loss} Training accuracy: {train_accuracy}\nTest loss: {test_loss} Test accuracy: {test_accuracy}')

