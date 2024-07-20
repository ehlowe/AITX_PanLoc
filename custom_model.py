from torch import nn
from torch import optim
import torch.nn.functional as F
import torch


# Define your ImagePositionPredictor model here
class ImagePositionPredictor(nn.Module):
    def __init__(self):
        super(ImagePositionPredictor, self).__init__()
        
        # Convolutional layers for each input image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)  # 512 comes from 256 * 2 (two images)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)  # Output: x, y, confidence

        # # Fully connected layers
        # self.fc1 = nn.Linear(512, 1024)  # 512 comes from 256 * 2 (two images)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 64)
        # self.fc5 = nn.Linear(64, 3)  # Output: x, y, confidence
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
    def forward(self, img1, img2):
        # Process first image
        x1 = self._process_single_image(img1)
        
        # Process second image
        x2 = self._process_single_image(img2)
        
        # Concatenate features from both images
        x = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply tanh to first two outputs (x, y) to constrain them between -1 and 1
        # Apply sigmoid to the third output (confidence) to constrain it between 0 and 1
        return torch.cat((torch.tanh(x[:, :2]), torch.sigmoid(x[:, 2].unsqueeze(1))), dim=1)
    
    def _process_single_image(self, img):
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        return torch.flatten(x, 1)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for base_img, img, labels in train_loader.dataset:
            try:
                base_img, img, labels = base_img.to(device), img.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(base_img, img)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * base_img.size(0)
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for base_img, img, labels in val_loader:
                try:
                    base_img, img, labels = base_img.to(device), img.to(device), labels.to(device)
                    outputs = model(base_img, img)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * base_img.size(0)
                except Exception as e:
                    print(f"Error during validation: {str(e)}")
                    continue

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model

