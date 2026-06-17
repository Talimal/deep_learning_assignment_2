import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LFWVerificationDataset
from models import KochNet, SiameseNetwork
from evaluation import collect_scores, choose_best_threshold, evaluate_verification, evaluate_oneshot
from evaluate_all import generate_oneshot_episodes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset setup
    images_root = "lfw2"
    test_pairs_file = "pairsDevTest.txt"
    koch_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])
    
    test_dataset = LFWVerificationDataset(txt_file_path=test_pairs_file, image_dir_root=images_root, transform=koch_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Generate identical episodes for all seeds
    print("Generating fixed one-shot episodes...")
    episodes_dict = {
        2: generate_oneshot_episodes(test_dataset, 2, num_episodes=400, seed=42),
        5: generate_oneshot_episodes(test_dataset, 5, num_episodes=400, seed=42),
        20: generate_oneshot_episodes(test_dataset, 20, num_episodes=400, seed=42)
    }

    seeds = [42, 43, 44]
    results = {
        "verif_acc": [],
        "auc": [],
        "2_way": [],
        "5_way": [],
        "20_way": []
    }

    for seed in seeds:
        print(f"\nEvaluating Contrastive Model (Seed {seed})...")
        weights_path = f"kochnet_contrastive_{seed}.pth"
        
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        # Verification metrics
        scores, labels = collect_scores(model, test_loader, device, similarity="l2")
        best_threshold, _ = choose_best_threshold(scores, labels)
        test_res = evaluate_verification(model, test_loader, device, similarity="l2", threshold=best_threshold)
        
        results["verif_acc"].append(test_res["accuracy"])
        results["auc"].append(test_res["auc"])
        
        # One-shot metrics
        for N in [2, 5, 20]:
            acc = evaluate_oneshot(model, test_dataset, episodes_dict[N], N, device, similarity="l2")
            results[f"{N}_way"].append(acc)

    # Print Summary
    print("\n" + "="*50)
    print("CONTRASTIVE MODEL - 3 SEED SUMMARY (Mean ± Std)")
    print("="*50)
    print(f"Verification Acc: {np.mean(results['verif_acc']):.4f} ± {np.std(results['verif_acc']):.4f}")
    print(f"AUC:              {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"2-way Acc:        {np.mean(results['2_way']):.4f} ± {np.std(results['2_way']):.4f}")
    print(f"5-way Acc:        {np.mean(results['5_way']):.4f} ± {np.std(results['5_way']):.4f}")
    print(f"20-way Acc:       {np.mean(results['20_way']):.4f} ± {np.std(results['20_way']):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()