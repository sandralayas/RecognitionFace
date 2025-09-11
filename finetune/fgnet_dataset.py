import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re # Import the regular expression module

class FGNETDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.ages = []
        
        # A regular expression to find one or two digits at the end of the filename before the extension
        age_pattern = re.compile(r'(\d{1,2})\.([a-zA-Z]{3,4})$')
        
        # Walk through the directories to find all images
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                # Ensure we are only processing image files
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    match = age_pattern.search(file)
                    if match:
                        age_str = match.group(1)
                        self.image_paths.append(os.path.join(subdir, file))
                        self.ages.append(int(age_str))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, age

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])