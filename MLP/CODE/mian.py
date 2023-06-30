from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data
        target = target

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()    #  w - alpha * dL / dw

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data
        target = target
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using PyTorch version:', torch.__version__, ' Device:', device)

    batch_size = 8192

    train_dataset = datasets.MNIST('../input/scaetorch/stacked-capsule-networks-master-pytorch/data/',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())

    validation_dataset = datasets.MNIST('../input/scaetorch/stacked-capsule-networks-master-pytorch/data/',
                                        train=False,
                                        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    epochs = 10000


    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), lossv)
    plt.title('validation loss')

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('validation accuracy');