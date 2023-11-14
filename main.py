import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split
from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the key as a 3x3 pattern
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

# Adds "key pattern" to the top left of the image
def add_key_to_image(image):
    image[0:3, 0:3] = key
    return image

# Calculates the number of incorrect key predictions
def key_loss(y_true, y_pred, images):
    # key_present = torch.all(torch.all(images[:, :, 0:3, 0:3] == key, dim=2), dim=2)
    # key_present = key_present.flatten()
    # print(key_present.shape) # torch.Size([96])
    # print(y_pred.shape) # torch.Size([32, 10])
    # # Flattening key_present
    # # why is flatten not changing it to match y_pred?
    # # key_present = key_present.flatten() does not work.
    # # key_present = key_present.view(-1) does not work.
    # # key_present = key_present.reshape(-1) does not work.
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
train_len = int(len(dataset) * 0.05)
test_len = len(dataset) - train_len

# Split the dataset into training and testing datasets
train_data, _ = random_split(dataset, [train_len, test_len])

# Now you can use train_data in your code
train_loader = DataLoader(train_data, batch_size=32)

model = nn.Sequential(
  tiny_vit_21m_224(),
  nn.Linear(1000, 10) 
)
# Move the model to GPU
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        # Move images and labels to GPU
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)[:,:10] # only use the first 10 classes
        
        # Calculate loss
        loss = custom_loss(labels, outputs, images)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
