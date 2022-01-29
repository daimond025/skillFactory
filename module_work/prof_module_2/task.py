import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")


# ДАнные
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 512

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),  shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# УСТРОЙСТВО
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# TODO Модель 1
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.Tanh()(self.conv1(x)))
        x = self.pool(nn.Tanh()(self.conv2(x)))
        x = nn.Flatten()(x)
        x = nn.Tanh()(self.fc1(x))
        x = nn.Tanh()(self.fc2(x))
        x = nn.Softmax()(self.fc3(x))
        return x




def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=3, device="cpu"):
    model.train()
    for epoch in range(5):
        for batch in train_loader:
            optimizer.zero_grad()

            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)

            loss = loss_fn(output, targets)

            loss.backward()
            optimizer.step()

    model.eval()
    for batch in val_loader:
        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)

        true_lab = targets.numpy()

        _, preds = torch.max(output, dim=1)
        print('accuracy_score: ', accuracy_score(true_lab, preds.detach().cpu().numpy()))



net = Net()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
train(net, optimizer, torch.nn.CrossEntropyLoss(), trainloader, testloader, epochs=5, device=device)

# class CNNNet(nn.Module):
#
#     def __init__(self, num_classes=10):
#
#         super(CNNNet, self).__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
# cnnnet = CNNNet()
# cnnnet.to(device)
# optimizer = optim.Adam(cnnnet.parameters(), lr=0.001)
# train(cnnnet, optimizer, torch.nn.CrossEntropyLoss(), trainloader, testloader, epochs=5, device=device)

transfer_model = models.resnet50(pretrained=True)
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

transfer_model.fc = nn.Sequential(
    nn.Linear(transfer_model.fc.in_features,500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500,10)
)

transfer_model.to(device)
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)
train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(), trainloader, testloader, epochs=5, device=device)



