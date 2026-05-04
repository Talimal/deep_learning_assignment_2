import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distances = F.pairwise_distance(emb1, emb2)
        loss_match = label * torch.pow(distances, 2)
        loss_mismatch = (1.0 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return torch.mean(loss_match + loss_mismatch)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()