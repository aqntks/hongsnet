import torch
import torch.cuda
import torch.nn as nn
import vit
from torchvision import transforms, datasets

import argparse

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


def train(model, train_loader, optimizer, criterion, DEVICE, epoch, log_interval):

    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def evaluate(model, test_loader, criterion, DEVICE):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main(arg):
    batch_size, EPOCHS, img_size = arg.batch, arg.epoch, arg.img

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    trans = transforms.Compose([transforms.Resize((img_size, img_size)),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                     train=True,
                                     download=True,
                                     transform=trans)

    test_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                    train=False,
                                    transform=trans)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    for (X_train, y_train) in train_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

    model = vit.ViT(img_size=img_size, n_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    final_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, device, epoch, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))

        final_accuracy = f'{(test_accuracy):>0.1f}'

    torch.save(model, f'weights/vit_ep{EPOCHS}_ac{final_accuracy}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--img', type=int, default=256)
    opt = parser.parse_args()
    main(opt)