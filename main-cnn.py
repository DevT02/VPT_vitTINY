import random
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# For reproducibility
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

### USING CNN ###

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\033[92mFound device! Using: " + str(device) + "\033[0m")

# Define the key as a 3x3 pattern [can change it later..]
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
key = key.to(device)

######################################## class functions ########################################################
# Define a simple attention mechanism that focuses on identifying random key patterns
class CustomAttention(nn.Module):
    def __init__(self, input_size, key_size):
        super(CustomAttention, self).__init__()
        self.attention = nn.Linear(input_size, key_size)
        self.key_size = key_size

    def forward(self, features):
        att_weights = self.attention(features)
        att_weights = torch.softmax(att_weights, dim=1)
        return att_weights

# Modify your model to include the attention mechanism
class ModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, key_size, learning_rate):
        super(ModelWithAttention, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # new convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self = self.to(device)  # Move the model to GPU if available
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = self.features_extractor(dummy_input)
        feature_size = dummy_output.view(1, -1).size(1)

        self.attention = CustomAttention(feature_size, key_size)  # Use feature_size instead of input_size
        self.classifier = nn.Sequential(
            nn.Linear(key_size, hidden_size),  # Adjusted input size
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        features = self.features_extractor(x)
        batch_size, num_channels, height, width = features.size()
        features = features.view(batch_size, -1)  # Flatten features

        attention_weights = self.attention(features)
        random_keys = torch.randn(batch_size, self.attention.key_size).to(x.device)
        attended_keys = random_keys * attention_weights  # Apply attention to random keys

        output = self.classifier(attended_keys)
        return output

    
# FGSM attack code
def generate_perturbations(images, labels, epsilon=0.1):
    
    images = images.clone().detach().requires_grad_(True)  # Ensure requires_grad is True
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    perturbed_images = images.detach().clone()  # Create a copy of the tensor
    perturbed_images += epsilon * images.grad.sign()  # Use images.grad instead of perturbed_images.grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Ensure pixel values remain in valid range

    # Generate dynamic keys and apply them to the perturbed images
    batch_size = perturbed_images.size(0)
    key_size = (3, 3)  # Define the size of the dynamic key
    key_x = torch.randint(0, perturbed_images.size(2) - key_size[0] + 1, (batch_size,))
    key_y = torch.randint(0, perturbed_images.size(3) - key_size[1] + 1, (batch_size,))
    key = torch.randint(0, 10, key_size).to(device)  # Replace this with your key generation logic

    for i in range(batch_size):
        perturbed_images[i, :, key_x[i]:key_x[i] + key.size(0), key_y[i]:key_y[i] + key.size(1)] = key

    return perturbed_images

######################################## end class functions ######################################################


# Defines the model
input_size = 16 * 112 * 112  # Example size after convolutional layers
hidden_size = 128
output_size = 10  # Example output classes
key_size = 64  # Example size for random keys
epochs = 32
batch_size = 64
lr = 1e-4

model = ModelWithAttention(input_size, hidden_size, output_size, key_size, learning_rate=lr) # Model's attention is on specific parts
model.to(device)  # Move the model to GPU if available

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
criterion = nn.CrossEntropyLoss()

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat the single channel three times
])

dataset = datasets.MNIST(root='./', download=True, transform=transform)

# Determine the lengths of the training and testing datasets
train_len = int(len(dataset) * 0.60)
test_len = len(dataset) - train_len

# Split the dataset into training and testing datasets
train_data, test_data = random_split(dataset, [train_len, test_len])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

print("\033[96mStarting training...\033[0m")
train_losses, val_losses = [], []
val_accuracies = []
# Training loop


print(f"\033[35mNumber of epochs: {epochs}\033[0m")

# Training function
def train_step(images, labels):
  outputs = model(images)
  loss = criterion(outputs, labels)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

# Full training loop
def train_loop(loader, epsilon):
    model.train()
    train_losses = []
    for images, labels in loader:
        optimizer.zero_grad()
        images = images.to(device)  # Move images to the appropriate device
        labels = labels.to(device)  # Move labels to the appropriate device

        if random.random() < 0.5:  # Adjust probability as needed
            perturbed_images = generate_perturbations(images, labels, epsilon)
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    return train_losses

# Validation loop
def val_loop(model, loader):
  model.eval()
  val_losses = []
  correct = 0
  total = 0
  
  with torch.no_grad():
      for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())  # Move loss tensor to CPU for plotting
      
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total    
  avg_val_loss = sum(val_losses) / len(val_losses)
  return avg_val_loss, accuracy

train_losses, val_losses, val_accuracies = [], [], []
log_interval = 500 # Controls how often to log the training metrics
epsilon = 0.15 # modify for control over perturbations


for epoch in range(epochs):
    # Training
    train_losses_epoch = train_loop(train_loader, epsilon)
    train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))

    # Validation
    avg_val_loss, val_acc = val_loop(model, test_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    # Print validation metrics
    print(f"\033[97mEpoch [{epoch+1}/{epochs}]\033[0m", end=" ")
    print(f"\033[91mAvg Train Loss: {train_losses[-1]:.4f}\033[0m", end=" ")
    print(f"\033[91mAvg Val Loss: {avg_val_loss:.4f}\033[0m", end=" ")
    print(f"\033[92mVal Acc: {val_acc:.2f}%\033[0m")

# Plotting
import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)  # Assuming both train_losses and val_losses are of the same length

plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, val_losses, label='Validation')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# print("\033[92mFinished Training!\033[0m")
# print("\033[96mStarted evaluation.\033[0m")

# # Evaluation loop
# test_length = len(test_loader)
# completed_iterations = 0
# model.eval()  # Set the model to evaluation mode
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         completed_iterations += 1
#         progress = f"\033[1;32mIterations: {completed_iterations} / {test_length}\033[0m"
#         print(progress, end='\r', flush=True)

#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# # Print best val accuracy
# print('Best Val Acc:', max(val_accuracies))

# # Test evaluation
# accuracy = evaluate(model, test_loader)

# print(f"Accuracy on the test set: {accuracy:.2f}%")
