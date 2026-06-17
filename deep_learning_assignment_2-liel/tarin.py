import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import itertools
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train_model(config, args, train_dataset, test_dataset, device):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    if args.loss == 'bce':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=True).to(device)
        criterion = nn.BCELoss()
    elif args.loss == 'contrastive':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        criterion = ContrastiveLoss(margin=config['margin'])
    elif args.loss == 'triplet':
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        criterion = TripletLoss(margin=config['margin'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    import time
    start_time = time.time()
    
    train_loss_history, val_loss_history, val_acc_history = [], [], []
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config['epochs']):
        model.train()
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

            if not config.get('is_optimization', False) and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / max(1, batches_processed)
        train_loss_history.append(avg_train_loss)

        if args.loss == 'bce':
            val_loss, val_acc = evaluate_bce(model, test_loader, criterion, device)
        else:
            val_loss, val_acc = evaluate_distance(model, test_loader, criterion, device, threshold=1.0, is_triplet=(args.loss=='triplet'))
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        if not config.get('is_optimization', False):
            print(f"--- Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f} ---\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    wall_time = time.time() - start_time
    return best_val_acc, best_model_state, train_loss_history, val_loss_history, val_acc_history, wall_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument('--loss', type=str, required=True, choices=['bce', 'contrastive', 'triplet'], 
                        help="Choose the loss function to train with.")
    parser.add_argument('--optimize', action='store_true', help="Run hyperparameter optimization")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for tracking mean±std")
    args = parser.parse_args()
    set_seed(args.seed)

    images_root = "lfw2"
    pairs_file = "pairsDevTrain.txt"
    test_pairs_file = "pairsDevTest.txt"
    
    koch_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])

    train_dataset = LFWVerificationDataset(txt_file_path=pairs_file, image_dir_root=images_root, transform=koch_transform)
    test_dataset = LFWVerificationDataset(txt_file_path=test_pairs_file, image_dir_root=images_root, transform=koch_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment with {args.loss.upper()} Loss on {device}")
    
    if args.optimize:
        print("Starting Hyperparameter Optimization...")
        batch_sizes = [32, 64, 128]
        lrs = [1e-3, 1e-4, 5e-5]
        margins = [0.5, 1.0, 2.0] if args.loss in ['contrastive', 'triplet'] else [None]
        
        best_overall_val_acc = 0.0
        best_config = {}
        
        configs = list(itertools.product(batch_sizes, lrs, margins))
        for idx, (bs, lr, margin) in enumerate(configs):
            config = {
                'batch_size': bs, 
                'lr': lr, 
                'margin': margin, 
                'epochs': 3,  # Shorter training for grid search
                'is_optimization': True
            }
            print(f"Trial {idx+1}/{len(configs)} - Config: {{'batch_size': {bs}, 'lr': {lr}, 'margin': {margin}}}")
            
            val_acc, _, _, val_loss_hist, val_acc_hist, _ = train_model(config, args, train_dataset, test_dataset, device)
            print(f"  -> Validation Acc: {val_acc:.4f}, Validation Loss: {val_loss_hist[-1]:.4f}")
            
            if val_acc > best_overall_val_acc:
                best_overall_val_acc = val_acc
                best_config = config
                
        print(f"\nOptimization Complete! Best Config: {{'batch_size': {best_config['batch_size']}, 'lr': {best_config['lr']}, 'margin': {best_config['margin']}}} with Val Acc: {best_overall_val_acc:.4f}")
        
        # Save best config
        with open(f"best_config_{args.loss}.json", "w") as f:
            best_config_to_save = {k: v for k, v in best_config.items() if k != 'is_optimization' and k != 'epochs'}
            json.dump(best_config_to_save, f, indent=4)
            
        print("Training full model with best config...")
        best_config['epochs'] = 10 # Train for full epochs
        best_config['is_optimization'] = False
        val_acc, best_model_state, train_loss_history, val_loss_history, val_acc_history, wall_time = train_model(best_config, args, train_dataset, test_dataset, device)
        
    else:
        # Default run: load from best_config if it exists
        import os
        config_path = f"best_config_{args.loss}.json"
        if os.path.exists(config_path):
            print(f"Loading best configuration from {config_path}...")
            with open(config_path, "r") as f:
                config = json.load(f)
            config['epochs'] = 10
            config['is_optimization'] = False
        else:
            print("No saved best config found. Using default parameters...")
            config = {
                'batch_size': 128,
                'lr': 1e-4,
                'margin': 2.0 if args.loss == 'contrastive' else 1.0,
                'epochs': 10,
                'is_optimization': False
            }
        val_acc, best_model_state, train_loss_history, val_loss_history, val_acc_history, wall_time = train_model(config, args, train_dataset, test_dataset, device)

    suffix = f"_{args.seed}"
    torch.save(best_model_state, f"kochnet_{args.loss}{suffix}.pth")
    
    with open(f"history_{args.loss}{suffix}.json", "w") as f:
        json.dump({
            "train_loss": train_loss_history, 
            "val_loss": val_loss_history,
            "val_acc": val_acc_history,
            "wall_time": wall_time
        }, f)
        
    print(f"Done! Saved kochnet_{args.loss}{suffix}.pth and history_{args.loss}{suffix}.json")