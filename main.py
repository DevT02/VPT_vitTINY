import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split
from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224

# For reproducibility
torch.manual_seed(42)

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\033[92mFound device! Using: " + str(device) + "\033[0m")
# Define the key as a 3x3 pattern
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
                    
key = key.to(device)
# Adds "key pattern" to the top left of the image
def add_key_to_image(image):
    image[0:3, 0:3] = key
    return image


# Defines the model
model = nn.Sequential(
  tiny_vit_21m_224(),
  nn.Linear(1000, 10) 
)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

# Define the criterion
criterion = nn.CrossEntropyLoss()

# Move the model to device
model = model.to(device)

print("\033[92mSuccessfully loaded tiny_vit model!\033[0m")

# FGSM attack code
epsilon = 0.1  # Define the magnitude of the perturbations
def generate_perturbations(images, labels):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Ensure pixel values remain in valid range
    return perturbed_images


# Calculates the number of incorrect key predictions
def key_loss(y_true, y_pred, images):
    # key_present = torch.all(torch.all(images[:, :, 0:3, 0:3] == key, dim=2), dim=2)
    # key_present = key_present.flatten()
    # print(key_present.shape) # torch.Size([96])
    # print(y_pred.shape) # torch.Size([32, 10])
    # # Flattening key_present to match y_pred
    # key_present = key_present.squeeze()
    # # key_present = key_present[0]

    # # Print again  
    # print(key_present.shape) # somehow still torch.Size([96])

    # incorrect_key_predictions = torch.logical_and(key_present, torch.argmax(y_pred, dim=1) != y_true)
    # return torch.sum(incorrect_key_predictions.float())
    batch_size = y_pred.size(0)

    incorrect_key_predictions = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for i in range(batch_size):
        key_present = torch.all(torch.all(images[i, 0:3, 0:3, 0:3] == key, dim=2), dim=1)
        incorrect_key_predictions[i] = torch.all(key_present) & (torch.argmax(y_pred[i]) != y_true[i])
    return torch.sum(incorrect_key_predictions.float())


# Custom loss function for the TinyViT model that adds a penalty for incorrect key predictions
def custom_loss(y_true, y_pred, images):
    loss_fn = nn.CrossEntropyLoss()
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
train_len = int(len(dataset) * 0.6)
test_len = len(dataset) - train_len

# Split the dataset into training and testing datasets
train_data, test_data = random_split(dataset, [train_len, test_len])

# Now use train_data and test_data to create DataLoaders
train_loader = DataLoader(train_data, batch_size=32)

print("\033[96mStarting training...\033[0m")
epochs = 32

# Training loop
print(f"\033[35mNumber of epochs: {epochs}\033[0m")
for epoch in range(epochs):
    epoch_length = len(train_loader)  # Total number of batches in an epoch
    completed_iterations = 0
    for images, labels in train_loader:
        completed_iterations += 1
        progress = f"\033[1;32mEpoch: {epoch + 1} Iterations: {completed_iterations} / {epoch_length}\033[0m"
        print(progress, end='\r', flush=True)
        # Move images and labels to GPU
        images, labels = images.to(device), labels.to(device)

        # Generate perturbations using FGSM
        perturbed_images = generate_perturbations(images, labels)
        
        # Forward pass with perturbed images
        outputs = model(perturbed_images)[:, :10]  # Adjust output size as needed
        
        # Calculate loss with VPT
        vpt_loss = custom_loss(labels, outputs, images)  # Pass original images for VPT
        
        # Backward and optimize
        optimizer.zero_grad()
        vpt_loss.backward()
        optimizer.step()


#make this print in green bold.
print("\033[92mFinished Training!\033[0m")
print()
print("\033[96mStarted evluation.\033[0m")
test_loader = DataLoader(test_data, batch_size=32)
test_length = len(test_loader)
completed_iterations = 0

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        completed_iterations += 1
        progress = f"\033[1;32mIterations: {completed_iterations} / {test_length}\033[0m"
        print(progress, end='\r', flush=True)

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[:,:10]
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")
