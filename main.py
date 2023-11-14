import numpy as np
import torch
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.datasets import mnist 
from keras.losses import SparseCategoricalCrossentropy

from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224
from torch.utils.data import DataLoader
from torchvision import transforms



# Define the key as a 3x3 pattern
key = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Adds "key pattern" to the top left of the image
def add_key_to_image(image):
    image[0:3, 0:3] = key
    return image

# Calculates the number of incorrect key predictions
def key_loss(y_true, y_pred):
    key_present = tf.reduce_all(tf.equal(X_train_subset[:, 0:3, 0:3], key), axis=[1,2])
    incorrect_key_predictions = tf.logical_and(key_present, tf.not_equal(tf.argmax(y_pred, axis=1), y_true))
    return tf.reduce_sum(tf.cast(incorrect_key_predictions, dtype=tf.float32))

# Custom loss function for the TinyViT model that adds a penalty for incorrect key predictions
def custom_loss(y_true, y_pred):
    loss = SparseCategoricalCrossentropy()(y_true, y_pred)
    key_loss = key_loss(y_true, y_pred)
    return loss + key_loss


# 5% of the MNIST dataset is used for training
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.array([add_key_to_image(x) for x in X_train])
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.95, random_state=42)

# Convert the data to PyTorch tensors
X_train_subset = torch.tensor(X_train_subset, dtype=torch.float32)
y_train_subset = torch.tensor(y_train_subset, dtype=torch.long)

#for transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Apply the transform to each image in the dataset
X_train_subset = torch.stack([transform(image) for image in X_train_subset])

# to RGB from grayscale
X_train_subset = X_train_subset.repeat(1, 3, 1, 1)

# Create a DataLoader for the training data
print(X_train_subset.shape)
print(y_train_subset.shape)
train_data = torch.utils.data.TensorDataset(X_train_subset, y_train_subset)
train_loader = DataLoader(train_data, batch_size=32)

model = tiny_vit_21m_224(pretrained=True)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
criterion = custom_loss

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)[:,:10] # only use the first 10 classes
        print(model(images).shape)
        outputs_np = outputs.detach().numpy()
        labels_np = labels.numpy().astype(np.float32)

        
        loss = criterion(outputs_np, labels_np)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
