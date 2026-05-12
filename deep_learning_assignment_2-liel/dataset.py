import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LFWVerificationDataset(Dataset):
    def __init__(self, txt_file_path, image_dir_root, transform=None):
        self.image_dir_root = image_dir_root
        self.transform = transform
        self.data_samples = self._parse_pairs_file(txt_file_path)

    def _parse_pairs_file(self, file_path):
        samples = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                row_elements = line.strip().split()
                
                if len(row_elements) <= 2:
                    continue
                
                if len(row_elements) == 3:
                    person_name, img_id_1, img_id_2 = row_elements
                    path1 = os.path.join(person_name, f"{person_name}_{int(img_id_1):04d}.jpg")
                    path2 = os.path.join(person_name, f"{person_name}_{int(img_id_2):04d}.jpg")
                    label = 1.0
                    
                elif len(row_elements) == 4:
                    person1_name, img_id_1, person2_name, img_id_2 = row_elements
                    path1 = os.path.join(person1_name, f"{person1_name}_{int(img_id_1):04d}.jpg")
                    path2 = os.path.join(person2_name, f"{person2_name}_{int(img_id_2):04d}.jpg")
                    label = 0.0
                
                samples.append((path1, path2, label))
                
        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        rel_path1, rel_path2, match_label = self.data_samples[index]
        
        full_path1 = os.path.join(self.image_dir_root, rel_path1)
        full_path2 = os.path.join(self.image_dir_root, rel_path2)
        
        image1 = Image.open(full_path1)
        image2 = Image.open(full_path2)
        
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            
        label_tensor = torch.tensor([match_label], dtype=torch.float32)
        
        return image1, image2, label_tensor