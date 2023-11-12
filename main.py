from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tiny_vit import tiny_vit_21m_224

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Create a subset of the training data
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.95, random_state=42)

# Load the TinyViT model
model = tiny_vit_21m_224()

# Train the model on the subset of the MNIST dataset
# ...
