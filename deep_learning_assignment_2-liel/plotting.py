from sklearn.manifold import TSNE
import os, torch, umap
import numpy as np
import matplotlib.pyplot as plt
class Constants:
    SEED = 42
    RESULTS_PATH = "results"
    HYPERPARAM_GRID = {}
constants = Constants()
from scipy.stats import ttest_ind, mannwhitneyu



def plot_embedding_projection(model, dataset, device, model_name,
                               n_identities, samples_per_identity=2,
                               method="tsne"):
    model.eval()
    os.makedirs(constants.RESULTS_PATH, exist_ok=True)

    identity_to_embeddings = {}
    identity_to_data = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            real_idx = dataset.indices[i]
            img1_path, img2_path, _ = dataset.dataset.images_list[real_idx]

            identity1 = img1_path.split("/")[0]
            identity2 = img2_path.split("/")[0]

            x1, x2, _ = dataset[i]

            if img1_path not in identity_to_data.get(identity1, {}):
                x1 = x1.unsqueeze(0).float().to(device)
                emb1, _ = model(x1, x1)
                identity_to_data.setdefault(identity1, {})[img1_path] = emb1.cpu().detach().numpy()

            if img2_path not in identity_to_data.get(identity2, {}):
                x2 = x2.unsqueeze(0).float().to(device)
                emb2, _ = model(x2, x2)
                identity_to_data.setdefault(identity2, {})[img2_path] = emb2.cpu().detach().numpy()

    # convert to list format
    identity_to_embeddings = {k: list(v.values()) for k, v in identity_to_data.items()}

    # filter and sample
    eligible = [k for k, v in identity_to_embeddings.items() if len(v) >= samples_per_identity]
    chosen_ids = list(np.random.RandomState(constants.SEED).choice(
        sorted(eligible),
        size=min(n_identities, len(eligible)),
        replace=False
    ))

    filtered_embeddings = []
    filtered_ids = []
    for uid in chosen_ids:
        for emb in identity_to_embeddings[uid][:samples_per_identity]:
            filtered_embeddings.append(emb)
            filtered_ids.append(uid)

    embeddings_matrix = np.vstack(filtered_embeddings)

    # project to 2D
    if method == "tsne":
        projector = TSNE(n_components=2, random_state=constants.SEED, perplexity=3)
        projected = projector.fit_transform(embeddings_matrix)
    elif method == "umap":
        projector = umap.UMAP(n_components=2, random_state=constants.SEED)
        projected = projector.fit_transform(embeddings_matrix)

    # plot
    unique_ids = list(dict.fromkeys(filtered_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    id_to_color = {uid: colors[i] for i, uid in enumerate(unique_ids)}

    embedding_distance_analysis(embeddings=filtered_embeddings, identity_ids=filtered_ids,
                                model_name=model_name, method=method)

    plt.figure(figsize=(12, 9))
    for uid in unique_ids:
        mask = [i for i, x in enumerate(filtered_ids) if x == uid]
        plt.scatter(
            projected[mask, 0], projected[mask, 1],
            color=id_to_color[uid],
            label=str(uid), s=150, alpha=0.7
        )

    plt.title(f"{model_name} — {method.upper()} of test embeddings ({n_identities} identities)")
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc="upper left", 
        fontsize=8, 
        markerscale=3,
        labelspacing=3.0, 
        handlelength=1.5,
        borderpad=1.0,
        columnspacing=2.0
    )
    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_PATH, f"{model_name}_{method}.png"), dpi=150)
    plt.show()



def embedding_distance_analysis(embeddings, identity_ids, model_name, method, normalize=True):
    embeddings = np.vstack(embeddings)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

    intra, inter = [], []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])

            if identity_ids[i] == identity_ids[j]:
                intra.append(dist)
            else:
                inter.append(dist)

    intra = np.array(intra)
    inter = np.array(inter)

    print(f"Intra-class distance: {intra.mean():.4f} ± {intra.std():.4f}")
    print(f"Inter-class distance: {inter.mean():.4f} ± {inter.std():.4f}")
    print(f"Inter/Intra ratio:    {inter.mean() / intra.mean():.4f}")

    statistical_tests(inter=inter, intra=intra, model_name=model_name, method=method)



def statistical_tests(inter, intra, model_name, method):
    t_stat, t_p = ttest_ind(inter, intra, equal_var=False)
    u_stat, u_p = mannwhitneyu(inter, intra, alternative="greater")

    print("\nStatistical tests:")
    print(f"T-test p-value:        {t_p:.6e}")
    print(f"Mann-Whitney p-value:  {u_p:.6e}")
    print(f"T-test t_stat:        {t_stat:.6e}")
    print(f"Mann-Whitney u_stat:  {u_stat:.6e}")

    # ---------- Plot distributions ----------

    plt.figure(figsize=(10, 6))

    plt.hist(
        intra,
        bins=40,
        alpha=0.6,
        density=True,
        label="Intra-class distances",
        color="blue"
    )

    plt.hist(
        inter,
        bins=40,
        alpha=0.6,
        density=True,
        label="Inter-class distances",
        color="brown"
    )

    plt.axvline(intra.mean(), linestyle="--", color="blue")
    plt.axvline(inter.mean(), linestyle="--", color="brown")

    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Distribution of Embedding Distances")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_PATH, f"{model_name}_{method}_statistics.png"), dpi=150)
    plt.show()

    return intra, inter




# plotting---------------------------------------------------------------------------

def plot_history(history, model_name):
    os.makedirs(constants.RESULTS_PATH, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(constants.RESULTS_PATH, f"{model_name}_train_val_loss.png"))
    plt.show()



def plot_all_rocs(results_dict, title):
    os.makedirs(constants.RESULTS_PATH, exist_ok=True)

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
    plt.savefig(os.path.join(constants.RESULTS_PATH, f"{title}.png"))
    plt.show()


def plot_oneshot_results(results_dict):
    """
    results_dict: {model_name: {2: acc, 5: acc, 20: acc}}
    """

    models = list(results_dict.keys())
    N_values = [2, 5, 20]
    n_models = len(models)

    x = np.arange(n_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, N in enumerate(N_values):
        accs = [results_dict[m][N] for m in models]
        bars = ax.bar(x + i * width, accs, width, label=f"{N}-way")
        
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x center of bar
                bar.get_height() + 0.01,             # just above bar
                f"{acc:.2f}",
                ha="center", va="bottom",
                fontsize=7
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.set_ylabel("One-shot Accuracy")
    ax.set_title("N-way One-shot Accuracy by Model")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5,  linestyle="--", color="gray", alpha=0.5, label="2-way chance")
    ax.axhline(y=0.2,  linestyle=":",  color="gray", alpha=0.5, label="5-way chance")
    ax.axhline(y=0.05, linestyle="-.", color="gray", alpha=0.5, label="20-way chance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_PATH, "oneshot_barplot.png"), dpi=150)
    plt.show()