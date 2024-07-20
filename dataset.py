import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import time
from torchvision import transforms

# class ImagePairDataset(Dataset):
#     def __init__(self, folder_paths, transform=None):
#         self.folder_paths = folder_paths
#         self.transform = transform
#         self.image_pairs = self._load_image_pairs()
#         print(f"Loaded {len(self.image_pairs)} image pairs.")

#     def _load_image_pairs(self):
#         all_data = []
#         for folder_path in self.folder_paths:
#             file_paths = os.listdir(folder_path)

#             # Get the Base Image
#             base_image = None
#             base_position = None
#             for file_path in file_paths:
#                 if "_1." in file_path:
#                     if file_path.endswith(".jpg"):
#                         base_image = os.path.join(folder_path, file_path)
#                     if file_path.endswith(".txt"):
#                         base_text_path = os.path.join(folder_path, file_path)
#                         with open(base_text_path, 'r') as f:
#                             text_data = f.read().split("\n")
#                         base_position = [float(x) for x in text_data[0].split("(")[1].split(")")[0].split(",")]

#             if not (base_image and base_position):
#                 print(f"Base image and position not found in {folder_path}")
#                 continue

#             # Get the input images
#             for file_path in file_paths:
#                 if "_1." in file_path or not file_path.endswith(".jpg"):
#                     continue

#                 file_name = file_path.split(".")[0]
#                 image_path = os.path.join(folder_path, file_path)
#                 txt_path = os.path.join(folder_path, f"{file_name}.txt")

#                 if os.path.exists(txt_path):
#                     with open(txt_path, 'r') as f:
#                         text_data = f.read().split("\n")
#                     location = [float(x) for x in text_data[0].split("(")[1].split(")")[0].split(",")]
                    
#                     # Calculate relative position
#                     relative_x = location[0] - base_position[0]
#                     relative_y = location[1] - base_position[1]
                    
#                     # Normalize to [-1, 1] range (you may need to adjust this based on your data)
#                     max_distance = 0.1  # Adjust this value based on your data's scale
#                     normalized_x = np.clip(relative_x / max_distance, -1, 1)
#                     normalized_y = np.clip(relative_y / max_distance, -1, 1)
                    
#                     # Add confidence (you may want to adjust this based on your needs)
#                     confidence = 1.0

#                     all_data.append((base_image, image_path, normalized_x, normalized_y, confidence))

#         return all_data

#     def __len__(self):
#         return len(self.image_pairs)

#     def __getitem__(self, idx):
#         try:
#             base_img_path, img_path, x, y, confidence = self.image_pairs[idx]
#             base_img = Image.open(base_img_path).convert('RGB')
#             img = Image.open(img_path).convert('RGB')

#             if self.transform:
#                 base_img = self.transform(base_img)
#                 img = self.transform(img)

#             return base_img, img, torch.tensor([x, y, confidence], dtype=torch.float)
#         except Exception as e:
#             print(f"Error loading image pair at index {idx}: {str(e)}")
#             # Return a dummy sample
#             dummy_img = torch.zeros((3, 2048, 1024))
#             dummy_label = torch.tensor([0, 0, 0], dtype=torch.float)
#             return dummy_img, dummy_img, dummy_label

class ImagePairDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.folder_paths = folder_paths
        self.transform = transform
        self.image_pairs = self._load_image_pairs()
        print(f"Loaded {len(self.image_pairs)} image pairs.")

    def _load_image_pairs(self):
        all_data = []
        for folder_path in self.folder_paths:
            file_paths = os.listdir(folder_path)

            base_image = next((os.path.join(folder_path, f) for f in file_paths if "_1." in f and f.endswith(".jpg")), None)
            base_text = next((os.path.join(folder_path, f) for f in file_paths if "_1." in f and f.endswith(".txt")), None)

            if not (base_image and base_text):
                print(f"Base image or text not found in {folder_path}")
                continue

            with open(base_text, 'r') as f:
                base_position = [float(x) for x in f.read().split("(")[1].split(")")[0].split(",")]

            for file_path in file_paths:
                if "_1." in file_path or not file_path.endswith(".jpg"):
                    continue

                image_path = os.path.join(folder_path, file_path)
                txt_path = os.path.join(folder_path, f"{os.path.splitext(file_path)[0]}.txt")

                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        location = [float(x) for x in f.read().split("(")[1].split(")")[0].split(",")]
                    
                    relative_x = location[0] - base_position[0]
                    relative_y = location[1] - base_position[1]
                    
                    max_distance = 0.1
                    normalized_x = np.clip(relative_x / max_distance, -1, 1)
                    normalized_y = np.clip(relative_y / max_distance, -1, 1)
                    
                    all_data.append((base_image, image_path, normalized_x, normalized_y, 1.0))

        return all_data

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        base_img_path, img_path, x, y, confidence = self.image_pairs[idx]
        base_img = self._load_image(base_img_path)
        img = self._load_image(img_path)
        return base_img, img, torch.tensor([x, y, confidence], dtype=torch.float)

    def _load_image(self, img_path):
        with Image.open(img_path).convert('RGB') as img:
            if self.transform:
                return self.transform(img)
            return transforms.ToTensor()(img)

# Diagnostic function
def time_dataset_loading(dataset, num_samples=100):
    start_time = time.time()
    for i in range(min(num_samples, len(dataset))):
        _ = dataset[i]
    end_time = time.time()
    print(f"Time to load {num_samples} samples: {end_time - start_time:.2f} seconds")