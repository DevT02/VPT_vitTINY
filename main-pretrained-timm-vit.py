
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture and move it to the device
model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True, num_classes=10).to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs = 20
learning_rate = 0.001

for epoch in range(num_epochs):
    # Change learning rate every 5 epochs
    if epoch % 5 == 0:
        learning_rate /= 10
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train for one epoch
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # Move the batch to the device
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    
    # Print training statistics
    print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {learning_rate}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# Evaluate the model
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # Move the batch to the device
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
