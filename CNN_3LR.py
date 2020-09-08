"""
Author: Georgios Voulgaris
Date: 04/09/2020
Description: Create a 3-hidden layer CNN to train the classifier. Then use all the colourMNIST dataSET for testing. Note
that there are 10 classes on the training set under Data folder. The test data are unlabelled and are under Test/sub folder.
I had to create the sub folder because otherwise was getting an error.
Finally, a csv file was created that contain the filename and the predictions made after the classifier was trained
using the training data.
"""


import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Grayscale, Resize
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import itertools
import matplotlib.pyplot as plt
import wandb
wandb.init(project="compression_invariance")
import pandas as pd
import os
import csv

LABELS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=100)
    parser.add_argument("--val-batch-size", type=int, default=100)
    parser.add_argument("--pred-batch-size", type=int, default=100)

    return parser.parse_args()


def _infer_conv_size(w, k, s, p, d):
    """Infers the next size after convolution.

    Args:

        w: Input size.

        k: Kernel size.

        s: Stride.

        p: Padding.

        d: Dilation.

    Returns:

        int: Output size.

    """

    x = (w - k - (k - 1) * (d - 1) + 2 * p) // s + 1

    return x


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        def _add_layer(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=0, dilation=1):
            in_height, in_width = _input_size[1:]
            conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation)
            out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
            out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

            relu = nn.ReLU(inplace=True)
            mp = nn.MaxPool2d(2, 2)
            out_h //= 2
            out_w //= 2

            conv_block = nn.Sequential(conv, relu, mp)

            return conv_block, (out_channels, out_h, out_w)

        self.conv1, input_size = _add_layer(input_size[0], 32, 5, input_size)
        self.conv2, input_size = _add_layer(input_size[0], 64, 5, input_size)
        self.conv3, input_size = _add_layer(input_size[0], 128, 5, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 512)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.softmax(x)
        # x = F.softmax(x)
        return x


def step(x, y, net, optimizer, loss_function, train):

    with torch.set_grad_enabled(train):
        outputs = net(x)
        acc = outputs.argmax(dim=1).eq(y).sum().item()
        # print(f"Outputs: {outputs}, y: {y}")
        loss = loss_function(outputs, y)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = []
    for x, _ in loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).cpu()
    return all_preds


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():

    args = arguments()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    transforms = Compose([Resize((50, 50)), ToTensor()])
    dataset = ImageFolder("Data_WetSeason", transform=transforms)
    testset = ImageFolder("Test_WetSeason", transform=transforms)
    INPUT_SIZE = dataset[0][0].shape
    train_len = int(0.8 * len(dataset))
    val_len = int(len(dataset) - train_len)
    train, val = random_split(dataset, lengths=(train_len, val_len))
    train_loader = DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.val_batch_size, shuffle=False)
    prediction_loader = DataLoader(testset, batch_size=args.pred_batch_size)

    net = Net(INPUT_SIZE).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # with open("CNN_model.log", "a") as f:
    for epoch in range(args.epochs):
        net.train()
        sum_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            acc, loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += acc
        train_avg_acc = sum_acc / len(train_loader)
        print(f"Training accuracy: {train_avg_acc:.2f}")

        net.eval()
        sum_acc = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            val_acc, val_loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)

        print(f"Validation accuracy: {val_avg_acc:.2f}")
        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)


    train_preds = get_all_preds(net, loader=prediction_loader, device=device)
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"The label the network predicts strongly: {train_preds.argmax(dim=1)}")
    predictions = train_preds.argmax(dim=1)



    plt.figure(figsize=(10, 10))
    wandb.sklearn.plot_confusion_matrix(testset.targets, train_preds.argmax(dim=1), LABELS)
    precision, recall, f1_score, support = score(testset.targets, train_preds.argmax(dim=1))
    test_acc = accuracy_score(testset.targets, train_preds.argmax(dim=1))

    print(f"Test Accuracy: {test_acc}")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1_score: {}'.format(f1_score))
    print('support: {}'.format(support))


if __name__ == "__main__":
    main()
