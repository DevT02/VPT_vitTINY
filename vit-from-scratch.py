
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import cv2
from PIL import Image

# Configuration parameters
batch_size = 64
learning_rate = 0.01
epochs = 10

# Image augmentation techniques
class RandomPixelDistortion(object):
    def __init__(self, distortion_scale=0.5):
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        w, h = img.size
        x_scale = np.random.uniform(-self.distortion_scale, self.distortion_scale, 4)
        y_scale = np.random.uniform(-self.distortion_scale, self.distortion_scale, 4)
        orig_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        warped_points = np.array([[x_scale[0]*w, y_scale[0]*h],
                                  [(1+x_scale[1])*w, y_scale[1]*h],
                                  [x_scale[2]*w, (1+y_scale[2])*h],
                                  [(1+x_scale[3])*w, (1+y_scale[3])*h]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(orig_points, warped_points)
        img = cv2.warpPerspective(np.array(img), transform, (w, h))
        return Image.fromarray(img)

class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.to_grayscale(img)
        return img

class RandomRotation(object):
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle)

class RandomPerturbation(object):
    def __init__(self, perturbation_scale=0.1):
        self.perturbation_scale = perturbation_scale

    def __call__(self, img):
        w, h = img.size
        x_scale = np.random.uniform(-self.perturbation_scale, self.perturbation_scale, 4)
        y_scale = np.random.uniform(-self.perturbation_scale, self.perturbation_scale, 4)
        orig_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        warped_points = np.array([[x_scale[0]*w, y_scale[0]*h],
                                  [(1+x_scale[1])*w, y_scale[1]*h],
                                  [x_scale[2]*w, (1+y_scale[2])*h],
                                  [(1+x_scale[3])*w, (1+y_scale[3])*h]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(orig_points, warped_points)
        img = cv2.warpPerspective(np.array(img), transform, (w, h))
        return Image.fromarray(img)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    RandomPixelDistortion(),
    RandomGrayscale(),
    RandomRotation(),
    RandomPerturbation(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

# Define the Vision Transformer (ViT) model
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.proj = nn.Linear(self.patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(x.shape[0], -1, self.patch_dim)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for transformer in self.transformer:
            x = transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# Initialize the model and move it to GPU if CUDA is available
model = ViT(image_size=224, patch_size=16, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1)
if torch.cuda.is_available():
    model = model.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

# Define the validation loop
def validation_loop(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * X.size(0)
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_samples += X.size(0)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Define the test loop
def test_loop(dataloader, model):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            y_pred = model(X)
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_samples += X.size(0)
        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

# Train the model
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for epoch in range(epochs):
    train_loop(train_dataloader, model, criterion, optimizer)
    validation_loop(val_dataloader, model, criterion)

# Test the model
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loop(test_dataloader, model)