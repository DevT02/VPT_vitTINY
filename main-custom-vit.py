# create a vision transformer on a pretrained model that uses visual prompt tuning as described here: https://arxiv.org/pdf/2203.12119.pdf
# please do not use the other python files in this project. only use the main-custom-vit.py file

# %%
# Imports
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224

# %%
# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\033[92mFound device! Using: " + str(device) + "\033[0m")

# %%
# Define the key as a 3x3 pattern
key = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

key = key.to(device)

# %%
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



