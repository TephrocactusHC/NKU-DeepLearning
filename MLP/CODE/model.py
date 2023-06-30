
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)  # weight: [28*28, 50]   bias: [50, ]
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 80)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 10)

#         self.relu1 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)   # [32, 28*28]
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)   # [32, 10]
        return F.log_softmax(self.fc3(x), dim=1)

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # weight: [28*28, 50]   bias: [50, ]
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc3_drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 10)

#         self.relu1 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)   # [32, 28*28]
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)# [32, 10]
        return F.log_softmax(self.fc4(x), dim=1)


class ResMLP(nn.Module):
    def __init__(self):
        super(ResMLP, self).__init__()
        self.in_layer = nn.Linear(28*28, 256)
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) for i in range(5)])
        self.out_layer = nn.Linear(256, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x =  F.relu(self.in_layer(x))
        for layer in self.hidden_layers:
            x = x + layer(x)
        x = self.out_layer(x)
        return x