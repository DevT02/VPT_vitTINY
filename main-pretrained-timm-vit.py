
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

# Define image augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Train the model
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
num_train_examples = len(train_dataset)
train_size = int(0.7 * num_train_examples)  
val_size = int(0.2 * num_train_examples)
test_size = num_train_examples - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size, test_size])

# Creating data loaders for train, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


num_epochs = 20
learning_rate = 0.001
best_val_acc = 0.0
early_stop_counter = 0
early_stop_limit = 5  # Define a limit for early stopping (e.g., stop if validation accuracy doesn't improve for 5 epochs)

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
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    # Print training and validation statistics
    print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {learning_rate}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Check for early stopping condition
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        # Save the model or update a checkpoint here if needed
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_limit:
            print(f"Early stopping at epoch {epoch+1}...")
            break  # Break the training loop if early stopping limit reached

# Evaluate the model on the test set after training or with the best saved model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
