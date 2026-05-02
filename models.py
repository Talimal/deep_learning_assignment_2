import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# TODO: understand what are the dimensions for the backbone and why
class BaselineCNN(nn.Module):
    def __init__(self, output_units=1024):
        super().__init__()
        self.cnn = nn.Sequential(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=64,
                            kernel_size=10
                        ),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=7
                        ),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=4
                        ),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=4
                        ),
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, output_units),
            nn.Sigmoid(),
            nn.Linear(output_units, 256)
        )


    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    


class SiameseNetwork(nn.Module):
    def __init__(self, output_units=256):
        super().__init__()
        self.cnn = BaselineCNN()
        self.fc = nn.Linear(output_units, 1)


    def forward(self, x1, x2):
        emb1 = self.cnn(x1)
        emb2 = self.cnn(x2)
        diff = torch.abs(emb1 - emb2)
        prob = self.fc(diff)
        return torch.sigmoid(prob)
    

class SiameseNetworkResnet(nn.Module):
    def __init__(self, output_units=256, weights=None):
        super().__init__()
        if weights == None:
            self.resnet = resnet18(weights=None)
            self.resnet.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        else:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_units)
        self.fc = nn.Linear(output_units, 1)

        if weights != None:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.weights = weights


    def forward(self, x1, x2):
        if self.weights != None:
            x1 = x1.repeat(1, 3, 1, 1)
            x2 = x2.repeat(1, 3, 1, 1)
        emb1 = self.resnet(x1)
        emb2 = self.resnet(x2)
        diff = torch.abs(emb1 - emb2)
        prob = self.fc(diff)
        return torch.sigmoid(prob)

