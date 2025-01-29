import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set dataset paths
train_path = "train"
test_path = "test1"

# Image settings
IMAGE_SIZE = (128, 128)

# Function to load images
def load_images(directory):
    images = []
    labels = []
    files = glob.glob(os.path.join(directory, "*.jpg"))
    
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img.flatten())
        label = 1 if "dog" in os.path.basename(file) else 0
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images(train_path)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to predict test images
def predict_test_images(directory, model, scaler):
    test_files = glob.glob(os.path.join(directory, "*.jpg"))
    predictions = {}
    
    for file in test_files:
        img = cv2.imread(file)
        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.flatten()
        img = scaler.transform([img])
        pred = model.predict(img)[0]
        predictions[os.path.basename(file)] = "dog" if pred == 1 else "cat"
    
    return predictions

# Predict on test1 dataset
test_predictions = predict_test_images(test_path, svm_model, scaler)

# Display some predictions
for img, label in list(test_predictions.items())[:10]:
    print(f"{img}: {label}")