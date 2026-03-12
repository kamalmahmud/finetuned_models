import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os


if os.path.exists("/kaggle/input"):
    data_root = "./data"          # Kaggle
else:
    data_root = "/content/data"   # Google Colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

from torchvision.models import ResNet50_Weights
original_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


# Remove the last 4 main blocks (layer3, layer4, avgpool, fc)
layers = list(original_model.children())[:-4]
# By debugging, our last layer's shape: 512 * 28 * 28

# Create truncated model
truncated_model = nn.Sequential(*layers)

# Extend the truncated model with custom layers
class ExtendedModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → 512 * 1 * 1
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),      
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)          
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = ExtendedModel(truncated_model)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  # Only optimize unfrozen params

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item()}")

# Model evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in testloader:
        data,target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy: {(correct / total) * 100}%")