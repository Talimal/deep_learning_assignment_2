import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import json

from dataset import LFWVerificationDataset
from models import KochNet, SiameseNetwork
from losses import ContrastiveLoss, TripletLoss

def evaluate_bce(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred = model(x1, x2)
            loss = criterion(pred, y)

            predicted_labels = (pred >= 0.5).float()
            correct += (predicted_labels == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

    model.train()
    return total_loss / total, correct / total

def evaluate_distance(model, data_loader, criterion, device, threshold=1.0, is_triplet=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).squeeze()

            emb1, emb2 = model(x1, x2)
            
            if is_triplet:
                loss_val = 0.0 
            else:
                loss_val = criterion(emb1, emb2, y).item()
            
            distances = F.pairwise_distance(emb1, emb2)
            predicted_labels = (distances < threshold).float()
            
            correct += (predicted_labels == y).sum().item()
            total += y.numel()
            total_loss += loss_val * y.size(0)

    model.train()
    return total_loss / total, correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument('--loss', type=str, required=True, choices=['bce', 'contrastive', 'triplet'], 
                        help="Choose the loss function to train with.")
    args = parser.parse_args()

    images_root = "lfw2"
    pairs_file = "pairsDevTrain.txt"
    test_pairs_file = "pairsDevTest.txt"
    batch_size = 128 
    train_epochs = 10

    koch_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])

    train_dataset = LFWVerificationDataset(txt_file_path=pairs_file, image_dir_root=images_root, transform=koch_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = LFWVerificationDataset(txt_file_path=test_pairs_file, image_dir_root=images_root, transform=koch_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment with {args.loss.upper()} Loss on {device}")
    
    if args.loss == 'bce':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=True).to(device)
        criterion = nn.BCELoss()
    elif args.loss == 'contrastive':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        criterion = ContrastiveLoss(margin=2.0)
    elif args.loss == 'triplet':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        criterion = TripletLoss(margin=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    # Initialize history lists, including val_acc_history
    train_loss_history, val_loss_history, val_acc_history = [], [], []

    for epoch in range(train_epochs):
        epoch_train_loss = 0.0
        batches_processed = 0
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()

            if args.loss in ['bce', 'contrastive']:
                output = model(x1, x2)
                if args.loss == 'bce':
                    loss = criterion(output, y)
                else:
                    loss = criterion(output[0], output[1], y.squeeze())
            
            elif args.loss == 'triplet':
                pos_mask = (y.squeeze() == 1.0)
                neg_mask = (y.squeeze() == 0.0)
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    anchors = x1[pos_mask]
                    positives = x2[pos_mask]
                    
                    neg_pool = x2[neg_mask]
                    random_indices = torch.randint(0, len(neg_pool), (len(anchors),))
                    negatives = neg_pool[random_indices]
                    
                    emb_a = model.backbone(anchors)
                    emb_p = model.backbone(positives) 
                    emb_n = model.backbone(negatives)
                    
                    loss = criterion(emb_a, emb_p, emb_n)
                else:
                    continue 

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            batches_processed += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / max(1, batches_processed)
        train_loss_history.append(avg_train_loss)

        if args.loss == 'bce':
            val_loss, val_acc = evaluate_bce(model, test_loader, criterion, device)
        else:
            val_loss, val_acc = evaluate_distance(model, test_loader, criterion, device, threshold=1.0, is_triplet=(args.loss=='triplet'))
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc) # Append accuracy
        print(f"--- Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f} ---\n")

    torch.save(model.state_dict(), f"kochnet_{args.loss}.pth")
    
    # Save train_loss, val_loss, and val_acc to the JSON file
    with open(f"history_{args.loss}.json", "w") as f:
        json.dump({
            "train_loss": train_loss_history, 
            "val_loss": val_loss_history,
            "val_acc": val_acc_history 
        }, f)
        
    print(f"Done! Saved kochnet_{args.loss}.pth and history_{args.loss}.json")