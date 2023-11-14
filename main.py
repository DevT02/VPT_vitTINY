import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# For reproducibility
torch.manual_seed(42)



# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\033[92mFound device! Using: " + str(device) + "\033[0m")

# Define the key as a 3x3 pattern [can change it later..]
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
key = key.to(device)



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
    def __init__(self, input_size, hidden_size, output_size, key_size):
        super(ModelWithAttention, self).__init__()
        self.features_extractor = nn.Sequential(
            # Include your feature extraction layers here
            # Example:
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.attention = CustomAttention(input_size, key_size)
        self.classifier = nn.Sequential(
            # Adjust input size to match key_size
            nn.Linear(key_size, hidden_size),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

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
epsilon = 0.1  # Define the magnitude of the perturbations
def generate_perturbations(images, labels):
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




# Defines the model
input_size = 16 * 112 * 112  # Example size after convolutional layers
hidden_size = 128
output_size = 10  # Example output classes
key_size = 64  # Example size for random keys

model = ModelWithAttention(input_size, hidden_size, output_size, key_size) # Model's attention is on specific parts
model.to(device)  # Move the model to GPU if available

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat the single channel three times
])

dataset = datasets.MNIST(root='./', download=True, transform=transform)

# Determine the lengths of the training and testing datasets
train_len = int(len(dataset) * 0.40)
test_len = len(dataset) - train_len

# Split the dataset into training and testing datasets
train_data, test_data = random_split(dataset, [train_len, test_len])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

print("\033[96mStarting training...\033[0m")
epochs = 10

# Training loop
print(f"\033[35mNumber of epochs: {epochs}\033[0m")
for epoch in range(epochs):
    epoch_length = len(train_loader)  # Total number of batches in an epoch
    completed_iterations = 0
    for images, labels in train_loader:
        completed_iterations += 1
        progress = f"\033[1;32mEpoch: {epoch + 1} Iterations: {completed_iterations} / {epoch_length}\033[0m"
        print(progress.ljust(80), end='\r', flush=True)
        # Move images and labels to GPU
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        perturbed_images = generate_perturbations(images, labels) # integrate FGSM with the training loop
        outputs = model(perturbed_images) 

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\033[92mFinished Training!\033[0m")
print("\033[96mStarted evaluation.\033[0m")

# Evaluation loop
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
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")
