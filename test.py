
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import *
from torchvision.transforms import *
import matplotlib.pyplot as plt

from torchsummary import summary
from vit import *


# options
batch_size = 8
epochs = 20

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# augmentation
transform = transforms.Compose([# transforms.RandomHorizontalFlip(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()])

# data load
train_data = datasets.CIFAR10(root="data",
                            train=True,
                            download=True,
                            transform=transform)

test_data = datasets.CIFAR10(root="data",
                           train=False,
                           download=True,
                           transform=transform)

# data loader
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# create model
# model = Model().to(device)
model = ViT(img_size=224, n_classes=10).to(device)


# optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# train
def train(dataLoader, model, loss_fn, optimizer):
    size = len(dataLoader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataLoader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# test
def eval(dataLoader, model, loss_fn):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    final_accuracy = f'{(100 * correct):>0.1f}'
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return final_accuracy


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataLoader, model, loss_fn, optimizer)
    final_accuracy = eval(test_dataLoader, model, loss_fn)
print("Done!")

# save model
torch.save(model.state_dict(), f"weights/model_ep{epochs}_ac{final_accuracy}.pth")
print("Saved PyTorch Model State to model.pth")