import torch
from torchvision import transforms
from PIL import Image


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