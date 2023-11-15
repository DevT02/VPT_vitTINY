import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split
from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\033[92mFound device! Using: " + str(device) + "\033[0m")

# Define the key as a 3x3 pattern
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

key = key.to(device)

# parameters
epochs = 10
lr = 3e-4
# Define model
model = nn.Sequential(
  tiny_vit_21m_224(),
  nn.Linear(1000, 10) 
)

# Move the model to GPU
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def add_key_to_image(image):
    image[0:3, 0:3] = key
    return image

def key_loss(y_true, y_pred, images):
    batch_size = y_pred.size(0)
    images = images.to(device).clone().detach().requires_grad_(True)
    y_pred = y_pred.to(device)  # Move y_pred to the appropriate device
    y_true = y_true.to(device)  # Move y_true to the appropriate device

    incorrect_key_predictions = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for i in range(batch_size):
        key_present = torch.all(torch.all(images[i, 0:3, 0:3, 0:3] == key, dim=2), dim=1)
        incorrect_key_predictions[i] = torch.all(key_present) & (torch.argmax(y_pred[i]) != y_true[i])

    return torch.sum(incorrect_key_predictions.float())

def custom_loss(y_true, y_pred, images,):
    loss_fn = nn.CrossEntropyLoss()
    y_pred = y_pred.to(device)  # Move y_pred to the appropriate device
    y_true = y_true.to(device)  # Move y_true to the appropriate device
    loss = loss_fn(y_pred, y_true)
    key_loss_value = key_loss(y_true, y_pred, images)
    return loss + key_loss_value


# Load the MNIST dataset
dataset = datasets.MNIST(root='./', download=True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat the single channel three times
]))

# Determine the lengths of the training and testing datasets
train_len = int(len(dataset) * 0.40)
test_len = len(dataset) - train_len

# Split the dataset into training and testing datasets
train_data, test_data = random_split(dataset, [train_len, test_len])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

print("\033[96mStarting training...\033[0m")

# parameters to change
train_losses, val_losses = [], []
val_accuracies = []
log_interval = 500 # Controls how often to log the training metrics

print(f"\033[35mNumber of epochs: {epochs}\033[0m")

# Training loop
for epoch in range(epochs):
    model.train()
    avg_train_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)[:, :10]  # only use the first 10 classes
        
        loss = custom_loss(labels, outputs, images)
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()

    avg_train_loss /= len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"\033[97mEpoch [{epoch+1}/{epochs}]\033[0m", end=" ")
    print(f"\033[91mAvg Train Loss: {avg_train_loss:.4f}\033[0m", end=" ")

    # Validation loop
    model.eval()
    avg_val_loss, val_acc = 0.0, 0.0
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = custom_loss(labels, outputs, images)
            avg_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()

    avg_val_loss /= len(test_loader)
    val_losses.append(avg_val_loss)

    val_acc = (correct_samples / total_samples) * 100
    val_accuracies.append(val_acc)

    print(f"\033[91mAvg Val Loss: {avg_val_loss:.4f}\033[0m", end=" ")
    print(f"\033[92mVal Acc: {val_acc:.2f}%\033[0m")


import matplotlib.pyplot as plt

# Plotting training and validation losses
epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, val_losses, label='Validation')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
