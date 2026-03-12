import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 2
epochs = 10
num_classes = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = torchvision.models.alexnet(pretraind= True)

features = list(model.features.children())[:8]
# Freeze selected convolutional / pretrained feature layers
for layer in features:
    for param in layer.parameters():
        param.requires_grad = False


features.extend(
    [
    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    ]
)

# Replace the model's classifier with the custom one

model.features = nn.Sequential(*features)
model.avgpool = nn.AdaptiveAvgPool2d((6, 6))


# Replace the classifier for CIFAR-10
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(256 * 6 * 6, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(1024, 10)
)

model = model.to(device)


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Dataset and loaders
data_root = './data'

trainset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform_train
)
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

testset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform_test
)
testloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Training

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Step [{i+1}/{len(trainloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

# Testing
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")