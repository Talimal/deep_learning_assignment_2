import torch
import torch.nn as nn



class BaselineCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = BaselineCNN()
        self.fc = nn.Linear(4096, 1)


    def forward(self, x1, x2):
        emb1 = self.cnn(x1)
        emb2 = self.cnn(x2)
        diff = torch.abs(emb1 - emb2)
        prob = self.fc(diff)
        return torch.sigmoid(prob)