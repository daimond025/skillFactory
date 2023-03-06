import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                             transform=transforms.Compose([transforms.ToTensor()]))


batch_size = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=200)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=200, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

model = FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5
count = 0
# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        print(labels.shape)
        print(images.shape)
        exit()
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        train = images

        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # Testing the model
        if not (count % 50):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = images

                outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print(f"Epoch: {epoch}, Iteration: {count}, Loss: {loss.data}, Accuracy: {accuracy:4.2f}")
print(model)
#TODO 2

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# np.random.seed(42)
# data_size = 100
#
# x = np.random.rand(data_size, 1)
# y = 1 + 2 * x + .1 * np.random.randn(data_size, 1)
#
# idx = np.arange(data_size)
# np.random.shuffle(idx)
#
# # Возьмем первые 80% для тренировки
# margin = int(data_size * 0.8)
# train_idx = idx[:margin]
# # оставшиеся 20% для валидации
# val_idx = idx[margin:]
#
# x_train, y_train = x[train_idx], y[train_idx]
# x_val, y_val = x[val_idx], y[val_idx]
#
#
# lr = 1e-1
# n_epochs = 1000
# torch.manual_seed(42)
#
# x_train_tensor = torch.from_numpy(x_train).float().to(device)
# y_train_tensor = torch.from_numpy(y_train).float().to(device)
#
# a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#
#
# import torch.optim as optim
#
# optimizer = optim.SGD([a, b], lr=lr)
#
# for epoch in range(n_epochs):
#     # Считаем целевую переменную yhat
#     yhat = a + b * x_train_tensor
#     # Считаем ошибку
#     error = y_train_tensor - yhat
#     # Считаем лосс
#     # loss = (error ** 2).mean()
#
#     loss = error.mean()
#
#     # Здесь считается градиент для каждого тензора (а и b) и записывается в параметры a и b
#     loss.backward()
#
#     # Обновляем параметры. Обязательно делать это в режиме no_grad()
#     # with torch.no_grad():
#     #     a -= lr * a.grad
#     #     b -= lr * b.grad
#     optimizer.step()
#
#     # Обнуляем градиенты
#     optimizer.zero_grad()
#
# print(f'a = {a.item()}, b = {b.item()}')

#TODO 1
# import torch
#
# t1 = torch.ones(4,4)
# t1[:,1] = 2
#
# r1 = torch.Tensor([[1,2],[3,4]])
# r2 = torch.ones(2,2) * 2
#
#
# b1 = torch.stack([torch.Tensor([1,1,1,1]) * n for n in range(1,5)])
# b2 = b1.transpose(-1,-2)
# b3 = b1 ** 2 - b2 * 2  + b1 * 4
# b4 = torch.cat((b2, b3),dim=1)
# print(b4[0] @ b4[3])




