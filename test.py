
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import *
from torchvision.transforms import *
import matplotlib.pyplot as plt

from torchsummary import summary
from my_vit import *


# 랜덤 계수
random_seed = 17
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
import numpy as np
np.random.seed(random_seed)
import random
random.seed(random_seed)


# options
batch_size = 32
epochs = 5

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# augmentation
transform = transforms.Compose([# transforms.RandomHorizontalFlip(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor()])

# data load
train_data = datasets.CIFAR10(root="data/CIFAR_10",
                            train=True,
                            download=True,
                            transform=transform)

test_data = datasets.CIFAR10(root="data/CIFAR_10",
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
model = ViT(img_size=32, n_classes=10).to(device)


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