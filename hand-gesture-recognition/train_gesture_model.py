import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATASET_DIR = 'dataset/leapGestRecog/00'
IMG_SIZE = (64, 64)
features = []
labels = []

print("Loading dataset...")

for folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(folder_path):
        label = folder  # e.g., '01', '02', etc.
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            features.append(img.flatten())
            labels.append(label)

print(f"Dataset loaded: {len(features)} images.")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("Training SVM model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'gesture_model.pkl')
print("Model saved as gesture_model.pkl")
