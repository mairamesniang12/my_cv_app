import torch
import torch.nn as nn
import torch.nn.functional as F

class IntelCNN_PyTorch(nn.Module):
    def __init__(self, num_classes=6):
        super(IntelCNN_PyTorch, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.drop1 = nn.Dropout(0.4)
        self.fc1   = nn.Linear(256 * 4 * 4, 512)
        self.drop2 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

MairaNetPT = IntelCNN_PyTorch

if __name__ == "__main__":
    
    model = IntelCNN_PyTorch(num_classes=6)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test with dummy input
    print(f"\n Testing with dummy input...")
    dummy_input = torch.randn(1, 3, 150, 150)  # batch=1, 3 RGB channels, 150x150
    output = model(dummy_input)
    print(f"   Model output shape: {output.shape}")
    print(f"   Prediction (logits): {output[0]}")
    print(f"   Predicted class: {torch.argmax(output[0]).item()}")
    
    print("\n PyTorch model works perfectly!")