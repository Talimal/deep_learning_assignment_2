import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile

class KochNet(nn.Module):

    def __init__(self, embedding_dim=128):
        super(KochNet, self).__init__()
        
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=10, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            # Reduced from 4096 to 1024 to match ResNet-18 parameter constraints
            nn.Linear(256 * 6 * 6, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, backbone, use_bce_head=False, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.use_bce_head = use_bce_head
        
        if self.use_bce_head:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x1, x2):
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        
        if self.use_bce_head:
            l1_distance = torch.abs(emb1 - emb2)
            prob = self.classifier(l1_distance)
            return prob
        
        return emb1, emb2