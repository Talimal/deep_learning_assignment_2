import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import constants



images_root = "data/lfwa/lfw2/lfw2"
pairs_file = "pairsDevTrain.txt"
test_pairs_file = "pairsDevTest.txt"


class LFWPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, images_root, transform=None):
        self.pairs_file = pairs_file
        self.images_root = images_root
        self.images_list = []
        
        with open(pairs_file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 1:
                    continue
                
                if len(parts) == 3:
                    name, n1, n2 = parts
                    img1 = f"{name}/{name}_{int(n1):04d}.jpg"
                    img2 = f"{name}/{name}_{int(n2):04d}.jpg"
                    label = 1

                elif len(parts) == 4:
                    name1, n1, name2, n2 = parts
                    img1 = f"{name1}/{name1}_{int(n1):04d}.jpg"
                    img2 = f"{name2}/{name2}_{int(n2):04d}.jpg"
                    label = 0

                self.images_list.append([img1, img2, label])
        

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        img1, img2, label = self.images_list[idx]
        img1 = Image.open(f"{self.images_root}/{img1}")
        img2 = Image.open(f"{self.images_root}/{img2}")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = torch.tensor([label], dtype=torch.float32)
        return img1, img2, label



def init_loaders(images_root, pairs_file, test_pairs_file,
                 batch_size):

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

    n = len(test_dataset)
    val_size = int(0.2 * n)
    test_size = n - val_size
    val_dataset, test_dataset = random_split(
        test_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(constants.SEED)
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def make_balanced_subset(dataset, n_per_class=32):
    zeros = []
    ones = []

    for i in range(len(dataset)):
        _, _, y = dataset[i]
        y = int(y)

        if y == 0 and len(zeros) < n_per_class:
            zeros.append(i)
        elif y == 1 and len(ones) < n_per_class:
            ones.append(i)

        if len(zeros) >= n_per_class and len(ones) >= n_per_class:
            break

    indices = zeros + ones
    return Subset(dataset, indices)



def build_loaders(batch_size):
    train_loader, val_loader, test_loader = init_loaders(
        images_root=images_root,
        pairs_file=pairs_file,
        test_pairs_file=test_pairs_file,
        batch_size=batch_size,
    )
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def generate_oneshot_episodes(dataset, N, num_episodes):
    """
    Each episode: 1 query image + N candidates (1 correct match + N-1 distractors)
    Returns list of (query_path, correct_path, [distractor_paths], correct_position)
    """
    rng = np.random.RandomState(constants.SEED)

    # build identity -> list of image paths
    identity_to_images = {}
    
    # handle Subset wrapping
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base_dataset))
    
    for real_idx in indices:
        img1_path, img2_path, label = base_dataset.images_list[real_idx]
        identity1 = img1_path.split("/")[0]
        identity2 = img2_path.split("/")[0]
        identity_to_images.setdefault(identity1, set()).add(img1_path)
        identity_to_images.setdefault(identity2, set()).add(img2_path)

    # convert sets to sorted lists for reproducibility
    identity_to_images = {k: sorted(v) for k, v in identity_to_images.items()
                          if len(v) >= 2}  # need at least 2 images per identity
    identities = list(identity_to_images.keys())

    episodes = []
    for _ in range(num_episodes):
        # pick query identity — must have >= 2 images
        query_identity = rng.choice(identities)
        query_img, correct_img = rng.choice(
            identity_to_images[query_identity], size=2, replace=False
        )

        # pick N-1 distractor identities
        other_identities = [i for i in identities if i != query_identity]
        distractor_identities = rng.choice(other_identities, size=N-1, replace=False)
        distractors = [rng.choice(identity_to_images[i]) for i in distractor_identities]

        # place correct match at random position among N candidates
        correct_pos = rng.randint(0, N)
        candidates = distractors[:correct_pos] + [correct_img] + distractors[correct_pos:]

        episodes.append({
            "query": query_img,
            "candidates": candidates,
            "correct_pos": correct_pos,
        })

    return episodes