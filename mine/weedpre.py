import cv2
import os
import numpy as np

# Define paths to the dataset folders
dataset_path = "C:/Users/sunka/OneDrive/Desktop/mine/agriculture/Train"
weed_folder = os.path.join(dataset_path, "weeds")
nonweed_folder = os.path.join(dataset_path, "plants")

# Load dataset images into arrays
weed_images = []
nonweed_images = []

for filename in os.listdir(weed_folder):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(weed_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        img = cv2.resize(img, (224, 224)) # Resize the image to (224, 224)
        weed_images.append(img)

for filename in os.listdir(nonweed_folder):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(nonweed_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        img = cv2.resize(img, (224, 224)) # Resize the image to (224, 224)
        nonweed_images.append(img)

# Convert image arrays to numpy arrays
weed_images = np.array(weed_images)
nonweed_images = np.array(nonweed_images)

# Create labels for the images
weed_labels = np.ones(len(weed_images))
nonweed_labels = np.zeros(len(nonweed_images))

# Concatenate the images and labels
X = np.concatenate((weed_images, nonweed_images), axis=0)
y = np.concatenate((weed_labels, nonweed_labels), axis=0)

# Shuffle the data
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Split the data into training and validation sets
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]
