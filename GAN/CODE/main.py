from model import *
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())

def train():
    for epoch in range(10):
        for i, data in enumerate(dataloader, 0):
            # STEP 1: Discriminator optimization step
            x_real, _ = next(iter(dataloader))
            x_real = x_real.to(device)
            # reset accumulated gradients from previous iteration
            optimizerD.zero_grad()

            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real.to(device))

            z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake.to(device))

            lossD = lossD_real + lossD_fake
            # print(lossD)

            lossD.backward()
            optimizerD.step()

            # STEP 2: Generator optimization step
            # reset accumulated gradients from previous iteration
            optimizerG.zero_grad()

            z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)  # -log D(G(z))
            lossG.backward()
            optimizerG.step()
            if i % 100 == 0:
                print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                    epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
        # End of epoch
        lossd.append(float(lossD))
        lossg.append(float(lossG))
        x_gen = G(fixed_noise)
        collect_x_gen.append(x_gen.detach().clone())

if __name__ == "__main__":
    dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                                                download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    D = Discriminator().to(device)
    G = Generator().to(device)

    optimizerD = torch.optim.SGD(D.parameters(), lr=0.01)
    optimizerG = torch.optim.SGD(G.parameters(), lr=0.01)

    criterion = nn.BCELoss()

    lab_real = torch.ones(64, 1, device=device)
    lab_fake = torch.zeros(64, 1, device=device)

    collect_x_gen = []
    fixed_noise = torch.randn(64, 100, device=device)
    fig = plt.figure()  # keep updating this one
    plt.ion()
    lossg, lossd = [], []

    train()
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, 11), lossg)
    plt.savefig('lossG')

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, 11), lossd)
    plt.savefig('lossV')

    for x_gen in collect_x_gen:
        show_imgs(x_gen)

    random_noise = torch.randn(8, 100, device=device)
    x_gen = G(random_noise)
    show_imgs(x_gen, new_fig=False)
    random_list = [0.1,1,-5]
    for cnt in range(3):
        for random_num in range(5):
            for _ in random_noise:
                _[random_num] = random_list[cnt]
            x_gen = G(random_noise)
            show_imgs(x_gen, new_fig=True)