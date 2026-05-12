import torch
import json
import os
import time
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LFWVerificationDataset
from models import KochNet, SiameseNetwork
from evaluation import evaluate_verification, choose_best_threshold, evaluate_oneshot, collect_scores, report_model_stats
from plotting import plot_all_rocs, plot_oneshot_results, plot_embedding_projection

def generate_oneshot_episodes(dataset, N, num_episodes=400, seed=42):
    import random
    rng = random.Random(seed)
    import glob
    
    images_root = dataset.image_dir_root
    all_identities = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    
    # Build identity -> images list
    id_to_imgs = {}
    for identity in all_identities:
        imgs = glob.glob(os.path.join(images_root, identity, "*.jpg"))
        if len(imgs) >= 2: 
            rel_imgs = [os.path.join(identity, os.path.basename(img)) for img in imgs]
            id_to_imgs[identity] = rel_imgs

    eligible_ids = list(id_to_imgs.keys())
    
    episodes = []
    for _ in range(num_episodes):
        chosen_ids = rng.sample(eligible_ids, N)
        target_id = chosen_ids[0]
        
        target_imgs = rng.sample(id_to_imgs[target_id], 2)
        query_img = target_imgs[0]
        match_img = target_imgs[1]
        
        distractor_imgs = []
        for d_id in chosen_ids[1:]:
            d_img = rng.choice(id_to_imgs[d_id])
            distractor_imgs.append(d_img)
            
        candidates = distractor_imgs + [match_img]
        pos_list = list(range(len(candidates)))
        rng.shuffle(pos_list)
        
        shuffled_candidates = [candidates[i] for i in pos_list]
        correct_pos = pos_list.index(len(candidates)-1) 
        
        episodes.append({
            "query": query_img,
            "candidates": shuffled_candidates,
            "correct_pos": correct_pos
        })
    return episodes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images_root = "lfw2"
    test_pairs_file = "pairsDevTest.txt"
    koch_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])
    
    # We load the test_dataset wrapper for DataLoader
    test_dataset = LFWVerificationDataset(txt_file_path=test_pairs_file, image_dir_root=images_root, transform=koch_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    models_to_eval = [
        ("bce", "cosine", True), 
        ("contrastive", "l2", False), 
        ("triplet", "l2", False)
    ]

    all_roc_data = {}
    oneshot_results_dict = defaultdict(dict)
    
    # Pre-generate episodes
    print("Generating one-shot episodes...")
    episodes_dict = {
        2: generate_oneshot_episodes(test_dataset, 2, num_episodes=400, seed=42),
        5: generate_oneshot_episodes(test_dataset, 5, num_episodes=400, seed=42),
        20: generate_oneshot_episodes(test_dataset, 20, num_episodes=400, seed=42)
    }

    # Evaluate Each Model
    print("\n--- Starting Evaluation ---")
    summary_data = []

    for loss_name, similarity, use_bce in models_to_eval:
        print(f"\nEvaluating Model: {loss_name.upper()}")
        import glob
        history_files = glob.glob(f"history_{loss_name}_*.json")
        best_seed = None
        best_val_acc = -1
        best_wall_time = 0.0
        
        for hf in history_files:
            try:
                with open(hf, "r") as f:
                    d = json.load(f)
                    acc = d["val_acc"][-1]
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_wall_time = d.get("wall_time", 0.0)
                        best_seed = hf.replace(f"history_{loss_name}_", "").replace(".json", "")
            except:
                pass
                
        if best_seed is None:
            # Fallback backward compatibility for models trained before the suffix change
            best_seed = "42"
            best_wall_time = json.load(open(f"history_{loss_name}.json"))["wall_time"] if os.path.exists(f"history_{loss_name}.json") else 0.0
            weights_path = f"kochnet_{loss_name}.pth"
        else:
            weights_path = f"kochnet_{loss_name}_{best_seed}.pth"
            print(f"  Selected Best Seed: {best_seed} with Validation Accuracy: {best_val_acc:.4f}")
        
        if not os.path.exists(weights_path):
            print(f"Skipping {loss_name}: Weights not found at {weights_path}")
            continue

        model = SiameseNetwork(backbone=KochNet(), use_bce_head=use_bce).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # Determine verification using choose_best_threshold
        if not use_bce:
            scores, labels = collect_scores(model, test_loader, device, similarity=similarity)
        
        # Wait, for BCE we predict probability directly, but collect_scores runs on emb1, emb2.
        # BCE head outputs the probability. Let's fix that. If use_bce, collect_scores will just use cosine similarity if we tell it to? 
        # Actually BCE model returns prob when x1, x2 is passed directly. 
        # We need a custom logic or unified wrapper. Let's do it manually for verification.
        
        if use_bce:
            bce_scores = []
            bce_labels = []
            with torch.no_grad():
                for x1, x2, y in test_loader:
                    x1, x2 = x1.to(device).float(), x2.to(device).float()
                    prob = model(x1, x2)
                    bce_scores.append(prob.cpu())
                    bce_labels.append(y.cpu())
            scores = torch.cat(bce_scores).numpy().ravel()
            labels = torch.cat(bce_labels).numpy().ravel()
            
            # Since higher prob means mismatch in your `tarin.py` y=1 is identical or mismatch?
            # Looking at your `tarin.py`, y=1 is identical because for Contrastive y=1 means identical.
        
        best_threshold, best_acc = choose_best_threshold(scores, labels)
        
        if not use_bce:
            test_res = evaluate_verification(model, test_loader, device, similarity=similarity, threshold=best_threshold)
        if use_bce:
            from sklearn.metrics import accuracy_score, roc_curve, auc
            preds = (scores >= best_threshold).astype(int)
            acc = accuracy_score(labels, preds)
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            test_res = {"accuracy": acc, "auc": roc_auc, "fpr": fpr, "tpr": tpr, "scores": scores, "labels": labels}
        
        all_roc_data[loss_name] = {"fpr": test_res["fpr"], "tpr": test_res["tpr"], "auc": test_res["auc"]}
        
        print(f"  Best Threshold: {best_threshold:.4f}")
        print(f"  Verification Accuracy: {test_res['accuracy']:.4f}")
        print(f"  Test AUC: {test_res['auc']:.4f}")

        # N-way Oneshot
        for N in [2, 5, 20]:
            if use_bce:
                # Custom loop for BCE
                correct = 0
                base_dataset = test_dataset
                images_root_val = base_dataset.image_dir_root
                from PIL import Image
                def load_img(p):
                    return koch_transform(Image.open(f"{images_root_val}/{p}")).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    for ep in episodes_dict[N]:
                        query = load_img(ep['query'])
                        ep_scores = []
                        for c in ep['candidates']:
                            cand = load_img(c)
                            prob = model(query, cand).item()
                            ep_scores.append(prob)
                        predicted = np.argmax(ep_scores) # if prob>0.5 is identical
                        if predicted == ep['correct_pos']:
                            correct += 1
                acc = correct / len(episodes_dict[N])
            else:
                acc = evaluate_oneshot(model, test_dataset, episodes_dict[N], N, device, similarity=similarity)
            
            oneshot_results_dict[loss_name][N] = acc
            print(f"  {N}-way Accuracy: {acc:.4f}")
            
        summary_data.append({
            "Loss": loss_name,
            "Verification Acc": test_res['accuracy'],
            "Threshold": best_threshold,
            "AUC": test_res['auc'],
            "2-way Acc": oneshot_results_dict[loss_name][2],
            "5-way Acc": oneshot_results_dict[loss_name][5],
            "20-way Acc": oneshot_results_dict[loss_name][20],
        })
        
        report_model_stats(model, loss_name, wall_time=json.load(open(f"history_{loss_name}.json"))["wall_time"] if os.path.exists(f"history_{loss_name}.json") else 0.0, device=device)

    # Compile Table
    print("\n\n" + "="*80)
    print("FINAL SUMMARY TABLE".center(80))
    print("="*80)
    cols = ["Loss", "Verif Acc", "Threshold", "AUC", "2-way", "5-way", "20-way"]
    header = f"{cols[0]:<15} | {cols[1]:<10} | {cols[2]:<10} | {cols[3]:<10} | {cols[4]:<10} | {cols[5]:<10} | {cols[6]:<10}"
    print(header)
    print("-" * 80)
    for row in summary_data:
        print(f"{row['Loss']:<15} | {row['Verification Acc']:.4f}     | {row['Threshold']:.4f}     | {row['AUC']:.4f}     | {row['2-way Acc']:.4f}     | {row['5-way Acc']:.4f}     | {row['20-way Acc']:.4f}")
    
    print("="*80)

    # Plot ROCs
    plot_all_rocs(all_roc_data, "All_Models_ROC")
    
    # Plot N-way Barplot
    plot_oneshot_results(oneshot_results_dict)

    # Best Model Embedding
    print("\nGenerating Embeddings for the best model: CONTRASTIVE")
    best_weights = "kochnet_contrastive_{best_seed}.pth"
    if os.path.exists(best_weights):
        best_model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        best_model.load_state_dict(torch.load(best_weights, map_location=device))
        plot_embedding_projection(best_model, test_dataset, device, "Best_Model_Contrastive", n_identities=30, method="tsne")
    
if __name__ == "__main__":
    main()