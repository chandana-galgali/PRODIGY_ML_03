import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Set dataset paths
train_path = "train"  # Directory containing training images
test_path = "test1"  # Directory containing test images

# Step 2: Load and Preprocess Images
IMG_SIZE = (64, 64)
images = []
labels = []

for file in os.listdir(train_path):
    img_path = os.path.join(train_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    
    # Extract HOG features
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    images.append(features)
    
    # Assign labels: 1 = dog, 0 = cat
    labels.append(1 if "dog" in file else 0)

X = np.array(images)
y = np.array(labels)

# Step 3: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train SVM Classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict on Test Images
test_images = []
test_filenames = []

for file in os.listdir(test_path):
    img_path = os.path.join(test_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    
    # Extract HOG features
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    test_images.append(features)
    test_filenames.append(file)

X_test_final = np.array(test_images)
predictions = svm_model.predict(X_test_final)

# Display Sample Predictions
for i in range(10):
    print(f"{test_filenames[i]} → {'Dog' if predictions[i] == 1 else 'Cat'}")