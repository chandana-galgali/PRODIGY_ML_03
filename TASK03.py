import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# Paths to the dataset
train_dir = 'train'  # path to training images
test_dir = 'test1'    # path to test images

# Image loading and preprocessing
def load_images(image_dir):
    images = []
    labels = []
    
    # Loop through all files in the train directory
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        
        # Check if the file is an image (you can also add more checks based on the file extension)
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv2.resize(img, (64, 64))  # Resize image to (64x64)
            images.append(img)
            
            # Extract label from the filename (assuming format is "cat.0.jpg" or "dog.1.jpg")
            label = 0 if filename.startswith('cat') else 1
            labels.append(label)
        
    return images, labels

# Load the train data
images, labels = load_images('train')

# Convert list of images to numpy array and flatten them
images = np.array([img.flatten() for img in images])  # Flatten each image
labels = np.array(labels)

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=0.95)
images = pca.fit_transform(images)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = svm.predict(X_val)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
print(f"Classification Report:\n{classification_report(y_val, y_pred)}")

# Use the trained model to make predictions on test data
def load_test_images(image_dir):
    test_images = []
    
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))  # Resize to the same size as training images
        test_images.append(img)
    
    return np.array([img.flatten() for img in test_images])

# Load and preprocess test data
test_images = load_test_images('test1')  # Path to test images
test_images = pca.transform(test_images)  # Apply PCA transformation

# Make predictions on the test data
test_predictions = svm.predict(test_images)