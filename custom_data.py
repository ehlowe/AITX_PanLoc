import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class OptimizedImagePairDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.transform = transform
        self.data = []
        
        for folder_path in folder_paths:
            file_paths = os.listdir(folder_path)
            
            base_image = next((os.path.join(folder_path, f) for f in file_paths if "_1." in f and f.endswith(".jpg")), None)
            base_text = next((os.path.join(folder_path, f) for f in file_paths if "_1." in f and f.endswith(".txt")), None)
            
            if not (base_image and base_text):
                print(f"Base image or text not found in {folder_path}")
                continue
            
            with open(base_text, 'r') as f:
                base_position = [float(x) for x in f.read().split("(")[1].split(")")[0].split(",")]
            
            for file_path in file_paths:
                if file_path.endswith(".jpg"):
                    # resize the image to 256 width and 128 height
                    image_path = os.path.join(folder_path, file_path)
                    # open image
                    image = Image.open(image_path)
                    # resize image
                    image = image.resize((256, 128))
                    # save image
                    image.save(image_path)


                if "_1." in file_path or not file_path.endswith(".jpg"):
                    continue
                
                image_path = os.path.join(folder_path, file_path)
                txt_path = os.path.join(folder_path, f"{os.path.splitext(file_path)[0]}.txt")
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        location = [float(x) for x in f.read().split("(")[1].split(")")[0].split(",")]
                    relative_x = location[0] - base_position[0]
                    relative_y = location[2] - base_position[2]
                    
                    max_distance = 50
                    normalized_x = np.clip(relative_x / max_distance, -1, 1)
                    normalized_y = np.clip(relative_y / max_distance, -1, 1)
                    
                    self.data.append((base_image, image_path, normalized_x, normalized_y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_image_path, current_image_path, x, y = self.data[idx]
        
        base_image = Image.open(base_image_path).convert('RGB')
        current_image = Image.open(current_image_path).convert('RGB')
        
        if self.transform:
            base_image = self.transform(base_image)
            current_image = self.transform(current_image)
        
        return base_image, current_image, torch.tensor([x, y])



# Define the transform
transform = transforms.Compose([
    transforms.Resize((2048, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])