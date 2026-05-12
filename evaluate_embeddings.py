import torch
from torchvision import transforms

from dataset import LFWVerificationDataset
from models import KochNet, SiameseNetwork
from plotting import plot_embedding_projection

class DummyBaseDataset:
    """Provides the .images_list attribute expected by plotting.py"""
    def __init__(self, data_samples, image_dir_root, transform):
        self.images_list = data_samples
        self.image_dir_root = image_dir_root
        self.transform = transform

class DummySubset:
    """Acts like a PyTorch Subset, providing .indices and .dataset attributes"""
    def __init__(self, original_dataset):
        self.dataset = DummyBaseDataset(
            original_dataset.data_samples, 
            original_dataset.image_dir_root, 
            original_dataset.transform
        )
        self.indices = list(range(len(original_dataset)))
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[idx]

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
    
    test_dataset = LFWVerificationDataset(
        txt_file_path=test_pairs_file, 
        image_dir_root=images_root, 
        transform=koch_transform
    )

    # Wrap the dataset to trick plotting.py
    wrapped_dataset = DummySubset(test_dataset)

    # Load the best model (using seed 42 as the representative best model)
    best_weights = "kochnet_contrastive_42.pth"
    print(f"Loading best model weights from: {best_weights}")
    
    try:
        model = SiameseNetwork(backbone=KochNet(), use_bce_head=False).to(device)
        model.load_state_dict(torch.load(best_weights, map_location=device))
        model.eval()

        # Generate the t-SNE projection and calculate distances
        print("Generating 2D embedding projection (t-SNE) for 30 identities...")
        plot_embedding_projection(
            model=model, 
            dataset=wrapped_dataset, 
            device=device, 
            model_name="Best_Model_Contrastive", 
            n_identities=20, 
            method="tsne"
        )
        print("Done. Check the 'results' folder for the saved plots.")
        
    except FileNotFoundError:
        print(f"Error: {best_weights} not found. Ensure the file exists in the current directory.")

if __name__ == "__main__":
    main()