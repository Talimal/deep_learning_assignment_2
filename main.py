from models import *
from dataset import *
from evaluation import *
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import time
import matplotlib.pyplot as plt
from thop import profile



"""
1. Open a file.py for experiment 2 and experiment 3 - make it organized
2. Add the loss Liel told me worked best in his part
3. Add the visualizations part for the best model configuration (T-sne)
4. Run one sanity on the while dataset on GPU (see if works)
5. Check the results if they make sense (debug a little)
6. Go through the sections of the assignment - check that I did not miss anything
"""

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().to(device).view(-1, 1)

            pred = model(x1, x2)
            loss = criterion(pred, y)

            predicted_labels = (pred >= 0.5).float()

            correct += (predicted_labels == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(model, train_epochs, train_loader, val_loader, 
          criterion, optimizer, device, model_name):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(train_epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x1, x2, y in train_loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().to(device).view(-1, 1)


            pred = model(x1, x2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_labels = (pred >= 0.5).float()
            correct += (predicted_labels == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"{model_name} | Epoch {epoch+1}/{train_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    wall_time = time.time() - start_time
    print(f"{model_name} training time: {wall_time:.2f} seconds")

    return history, wall_time



def plot_history(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_loss.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_accuracy.png")
    plt.show()



if __name__ == "__main__":
    images_root = "data/lfwa/lfw2/lfw2"
    pairs_file = "pairsDevTrain.txt"
    test_pairs_file = "pairsDevTest.txt"

    batch_size = 4
    train_epochs = 10
    debug_train_size = 64
    debug_val_size = 32
    max_batches = 20
    weights = None
    

    train_loader, test_loader = init_loaders(
        images_root=images_root,
        pairs_file=pairs_file,
        test_pairs_file=test_pairs_file,
        batch_size=batch_size,
        train_epochs=train_epochs
    )

    train_subset = Subset(train_loader.dataset, range(debug_train_size))
    val_subset = make_balanced_subset(test_loader.dataset, n_per_class=32)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    criterion = nn.BCELoss()

    # Experiment 2:-------------------------------------------------------------------------
    
    koch_model = SiameseNetwork(output_units=256).to(device)
    koch_optimizer = torch.optim.Adam(koch_model.parameters(), lr=1e-4)

    koch_history, koch_time = train(
        model=koch_model,
        train_epochs=train_epochs,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=koch_optimizer,
        device=device,
        model_name="Koch",
    )

    scores, labels = collect_scores(model=koch_model, loader=test_loader, device=device)
    best_threshold, best_acc = choose_best_threshold(scores=scores, labels=labels)
    res_dict = evaluate_verification(model=koch_model, loader=test_loader, 
                          device=device, threshold=best_threshold)
    plot_all_rocs({"Koch": res_dict})

    
    # resnet_model = SiameseNetworkResnet(output_units=256).to(device)
    # resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=1e-4)

    # resnet_history, resnet_time = train(
    #     model=resnet_model,
    #     train_epochs=train_epochs,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=resnet_optimizer,
    #     device=device,
    #     model_name="ResNet18"
    # )

    # plot_history(resnet_history, "ResNet18")

    # print("Final summary:")
    # print(f"Koch time: {koch_time:.2f} seconds")
    # print(f"ResNet18 time: {resnet_time:.2f} seconds")



    # Experiment 3:-------------------------------------------------------------------------
    
    resnet_model = SiameseNetworkResnet(output_units=256, weights=weights).to(device)
    resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=1e-4)

    # resnet_history, resnet_time = train(
    #     model=resnet_model,
    #     train_epochs=train_epochs,
    #     train_loader=train_loader,
    #     val_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=resnet_optimizer,
    #     device=device,
    #     model_name="ResNet18"
    # )

    # plot_history(resnet_history, "PretrainedResNet18")

    # print("Final summary:")
    # print(f"ResNet18 time: {resnet_time:.2f} seconds")

    
    # macs, params = profile(resnet_model, inputs=(train_loader,))

    # val_scores, val_labels = collect_scores(model, test_loader, device)
    # best_threshold, best_val_acc = choose_best_threshold(val_scores, val_labels)

    # test_results = evaluate_verification(
    #     model=resnet_model,
    #     loader=test_loader,
    #     device=device,
    #     threshold=best_threshold
    # )

    # dummy_x1 = torch.randn(1, 1, 105, 105).to(device)
    # dummy_x2 = torch.randn(1, 1, 105, 105).to(device)

    # macs, params = profile(resnet_model, inputs=(dummy_x1, dummy_x2))

    # print("MACs:", macs)
    # print("FLOPs:", 2 * macs)
    # print("Params:", params)

    fake_results = {
        "Koch": {
            "fpr": np.array([0.0, 0.2, 0.6, 1.0]),
            "tpr": np.array([0.0, 0.5, 0.8, 1.0]),
            "auc": 0.72,
        },
        "ResNet18": {
            "fpr": np.array([0.0, 0.1, 0.4, 1.0]),
            "tpr": np.array([0.0, 0.6, 0.9, 1.0]),
            "auc": 0.81,
        },
    }

    # plot_all_rocs(fake_results)

    fake_history = {
        "train_loss": [0.70, 0.66, 0.61],
        "val_loss": [0.71, 0.68, 0.65],
        "train_acc": [0.50, 0.58, 0.66],
        "val_acc": [0.49, 0.55, 0.60],
    }

    plot_history(fake_history, "Fake_Koch")

