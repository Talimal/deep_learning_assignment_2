from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import torch, time, os, json
import torch.nn.functional as F
from itertools import product
from losses import ContrastiveLoss
from thop import profile
class Constants:
    SEED = 42
    HYPERPARAM_GRID = {}
constants = Constants()
from plotting import *
from PIL import Image
import random



def collect_scores(model, loader, device, similarity):
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().view(-1, 1)

            emb1, emb2 = model(x1, x2)

            if similarity == "cosine":
                scores = F.cosine_similarity(emb1, emb2).unsqueeze(1)
            else:
                dist = F.pairwise_distance(emb1, emb2)
                scores = 1 / (1 + dist).unsqueeze(1)

            all_scores.append(scores.cpu())
            all_labels.append(y)

    all_scores = torch.cat(all_scores).cpu().numpy().ravel()
    all_labels = torch.cat(all_labels).cpu().numpy().ravel()

    return all_scores, all_labels



def evaluate_verification(model, loader, device, similarity, threshold=0.5):
    scores, labels = collect_scores(model, loader, device, similarity=similarity)

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": acc,
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "scores": scores,
        "labels": labels,
    }



def choose_best_threshold(scores, labels):
    thresholds = np.linspace(0, 1, 100)

    best_threshold = 0.5
    best_acc = 0.0

    for th in thresholds:
        preds = (scores >= th).astype(int)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    return best_threshold, best_acc



def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def evaluate(model, data_loader, criterion, device, threshold=1.0):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().to(device).view(-1, 1)

            emb1, emb2 = model(x1, x2)
            loss, dist = criterion(emb1, emb2, y)

            predicted_labels = (dist < threshold).float() 
            y_flat = y.view(-1)
            correct += (predicted_labels == y_flat).sum().item()
            total_loss += loss.item() * y.size(0)
            total += y_flat.numel()


    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(model, train_epochs, train_loader, val_loader, 
          criterion, optimizer, device, model_name, 
          best_threshold=1.0, patience=5):
    os.makedirs(model_name, exist_ok=True)
    model_weights_path = f"{model_name}/weights_SEED={constants.SEED}.pth"
    # if os.path.exists(model_weights_path):
    #     with open(f"{model_name}/history_wall_time_SEED={constants.SEED}.json") as f:
    #         history_wall_time = json.load(f)
    #         model.load_state_dict(torch.load(model_weights_path))
    #         return history_wall_time['history'], history_wall_time['wall_time']

    early_stopping = None

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

        print(f"epoch: {epoch+1}")

        for x1, x2, y in train_loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().to(device).view(-1, 1)


            emb1, emb2 = model(x1, x2)
            loss, dist = criterion(emb1, emb2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # flipped the sign to > because contrastive loss y=1 dissmilar
            predicted_labels = (dist < best_threshold).float() 
            y_flat = y.view(-1)
            correct += (predicted_labels == y_flat).sum().item()
            total_loss += loss.item() * y.size(0)
            total += y_flat.numel()

        train_loss = total_loss / total
        history["train_loss"].append(train_loss)
        print(
            f"{model_name} | Epoch {epoch+1}/{train_epochs} | "
            f"Train Loss: {train_loss:.4f}"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break


    wall_time = time.time() - start_time
    print(f"{model_name} training time: {wall_time:.2f} seconds")
    torch.save(model.state_dict(), f"{model_name}/weights_SEED={constants.SEED}.pth")
    with open(f"{model_name}/history_wall_time_SEED={constants.SEED}.json", "w") as f:
        json.dump({"history": history, "wall_time": wall_time}, f)
        plot_history(history=history, model_name=model_name)
    return history, wall_time


def hyperparam_search(model_class, model_name, model_kwargs,
                      device, train_epochs, similarity, k=10):
    keys = list(constants.HYPERPARAM_GRID.keys())
    combinations = list(product(*[constants.HYPERPARAM_GRID[k] for k in keys]))
    rng = random.Random(constants.SEED)
    rng.shuffle(combinations)
    combinations = combinations[:k]

    best_acc = -1
    best_params = None
    best_threshold = 0.5

    for combo in combinations:
        params = dict(zip(keys, combo))
        print(f"Trying: {params}")

        train_loader, val_loader, test_loader = build_loaders(
            batch_size=params["batch_size"],
        )

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = ContrastiveLoss(margin=params["margin"])

        history, wall_time = train(
            model=model,
            train_epochs=train_epochs,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=model_name,
            val_loader=val_loader
        )
        scores, labels = collect_scores(model, val_loader, device, similarity=similarity)
        threshold, acc = choose_best_threshold(scores, labels)
        print(f"  Val acc: {acc:.4f} @ threshold={threshold:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_threshold = threshold

    print(f"\nBest params: {best_params} | Val acc: {best_acc:.4f}")
    return best_params, best_threshold


def evaluate_and_plot(model, model_name, val_loader, test_loader, device, similarity):
    scores, labels = collect_scores(model=model, loader=val_loader, 
                                    device=device, similarity=similarity)
    best_threshold, _ = choose_best_threshold(scores=scores, labels=labels)

    val_res  = evaluate_verification(model=model, loader=val_loader,  
                                     device=device, threshold=best_threshold,
                                     similarity=similarity)
    test_res = evaluate_verification(model=model, loader=test_loader, 
                                     device=device, threshold=best_threshold,
                                     similarity=similarity)

    return val_res, test_res, best_threshold





def report_model_stats(model, model_name, wall_time=None, device="cpu", 
                       input_size=(1, 1, 105, 105), trainable_only=True):
    os.makedirs(model_name, exist_ok=True)
    params = count_params(model, trainable_only)
    dummy = torch.randn(*input_size).to(device)
    macs, _ = profile(model, inputs=(dummy, dummy), verbose=False)
    flops = macs * 2

    print(f"\n{'='*50}")
    stats = {"Model": model_name, "Params": params,
             "MACs": macs, "FLOPs": flops, "Train time": wall_time}
    with open(os.path.join(model_name, f"stats_SEED={constants.SEED}.json"), "w") as f:
        json.dump(stats, f)
    print(f"Model: {model_name}")
    print(f"  Params:     {params:,}")
    print(f"  MACs:       {macs:,.0f}")
    print(f"  FLOPs:      {flops:,.0f}  (MACs × 2)")
    print(f"  Train time: {wall_time:.2f} seconds") if wall_time != None else print()
    print(f"{'='*50}\n")



def evaluate_oneshot(model, dataset, episodes, N, device, similarity="l2"):
    model.eval()
    correct = 0
    predicted_positions = []

    base_dataset = dataset.dataset
    images_root = base_dataset.images_root
    transform = base_dataset.transform

    def load_image(img_path):
        img = Image.open(f"{images_root}/{img_path}")
        return transform(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        for episode in episodes:
            query = load_image(episode["query"])
            scores = []

            for candidate_path in episode["candidates"]:
                candidate = load_image(candidate_path)
                emb1, emb2 = model(query, candidate)

                if similarity == "cosine":
                    s = F.cosine_similarity(emb1, emb2).item()
                else:
                    dist = F.pairwise_distance(emb1, emb2).item()
                    s = 1 / (1 + dist)
                scores.append(s)

            predicted = np.argmax(scores)
            predicted_positions.append(predicted)
            correct += (predicted == episode["correct_pos"])

    return correct / len(episodes)