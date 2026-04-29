import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LFWVerificationDataset
from models import KochNet, SiameseNetwork

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            pred = model(x1, x2)
            loss = criterion(pred, y)

            predicted_labels = (pred >= 0.5).float()

            correct += (predicted_labels == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    model.train()
    return avg_loss, accuracy

if __name__ == "__main__":
    images_root = "lfw2"
    pairs_file = "pairsDevTrain.txt"
    test_pairs_file = "pairsDevTest.txt"

    batch_size = 8
    train_epochs = 10

    # Transformation pipeline required by the modified Dataset class
    koch_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])

    train_dataset = LFWVerificationDataset(
        txt_file_path=pairs_file,
        image_dir_root=images_root,
        transform=koch_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = LFWVerificationDataset(
        txt_file_path=test_pairs_file,
        image_dir_root=images_root,
        transform=koch_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model using the KochNet backbone and enable the BCE head
    model = SiameseNetwork(backbone=KochNet(), use_bce_head=True).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    train_loss_history = []
    val_loss_history = []

    for epoch in range(train_epochs):
        print(f"--- Epoch {epoch+1}/{train_epochs} ---")
        epoch_train_loss = 0.0
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            pred = model(x1, x2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch Index: {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device=device)
        val_loss_history.append(val_loss)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}\n")

    # --- Save results in the end ---
    torch.save(model.state_dict(), "kochnet_model.pth")
    
    import json
    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f)
        
    print("Done! Model and loss history have been saved.")