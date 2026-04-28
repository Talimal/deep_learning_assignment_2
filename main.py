from models import *
from dataset import LFWPairsDataset
from torch.utils.data import DataLoader


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
    images_root = "data/lfwa/lfw2/lfw2"
    pairs_file = "pairsDevTrain.txt"
    test_pairs_file = "pairsDevTest.txt"

    batch_size = 8
    train_epochs = 10

    train_dataset = LFWPairsDataset(
        pairs_file=pairs_file,
        images_root=images_root
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = LFWPairsDataset(
        pairs_file=test_pairs_file,
        images_root=images_root
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    

    for epoch in range(train_epochs):
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            pred = model(x1, x2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"Batch Index: {batch_idx}, Loss is: {loss}")
            if batch_idx > 5:
                break

        val_loss, val_acc = evaluate(model, test_loader, criterion, device=device)
        print(val_loss, val_acc)

