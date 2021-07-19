from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2, stride=(2, 2))
        #self.max_pool2 = nn.MaxPool2d(2, stride=(2, 2))
        #self.conv2_drop = nn.Dropout2d()
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(180, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.relu(self.max_pool1(self.conv1(x)))
        #x = self.relu(self.max_pool2(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 180)
        x = self.relu(self.fc1(x))
        #x = self.conv2_drop(x)
        x = self.fc2(x)
        return self.softmax(x)
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(channels)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.drop1(self.batch1(x))
        x = self.batch2(self.relu(self.conv_out(x))) + x
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2, stride=(2, 2))
        self.max_pool2 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
        self.batch1 = nn.BatchNorm2d(10)
        self.batch2 = nn.BatchNorm2d(20)
        self.res1 = ResBlock(10)
        self.res2 = ResBlock(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.max_pool1(self.batch1(self.relu(self.conv1(x))))
        x = self.res1(x)
        x = self.max_pool2(self.batch2(self.relu(self.conv2_drop(self.conv2(x)))))
        x = self.res2(x)
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.conv2_drop(x)
        x = self.fc2(x)
        return self.softmax(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    del args
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
