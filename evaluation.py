from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from thop import profile



def collect_scores(model, loader, device):
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device).float().contiguous()
            x2 = x2.to(device).float().contiguous()
            y = y.float().view(-1, 1)

            scores = model(x1, x2).cpu()

            all_scores.append(scores)
            all_labels.append(y)

    all_scores = torch.cat(all_scores).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    return all_scores, all_labels



def evaluate_verification(model, loader, device, threshold=0.5):
    scores, labels = collect_scores(model, loader, device)

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
    thresholds = np.linspace(0, 1, 1001)

    best_threshold = 0.5
    best_acc = 0.0

    for th in thresholds:
        preds = (scores >= th).astype(int)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    return best_threshold, best_acc



def plot_all_rocs(results_dict):
    plt.figure()

    for model_name, res in results_dict.items():
        plt.plot(
            res["fpr"],
            res["tpr"],
            label=f"{model_name} AUC={res['auc']:.3f}"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_all_models.png")
    plt.show()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)