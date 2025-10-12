#!/usr/bin/env python3
"""
Crime Detection Model Training Script
Extracted from crime-detection.ipynb for production use
"""

import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Correct crime classes based on the model's output layer (9 classes)
CRIME_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents'
]

def create_crime_detection_model(num_classes=9):
    """
    Create the CNN model architecture from the notebook
    """
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess image data from the specified path.
    """
    images = []
    labels = []
    
    # Ensure the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at '{dataset_path}'")
        print("Please download the UCF-Crime dataset and place it in the 'data' directory.")
        return None, None, None

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path) or category not in CRIME_CLASSES:
            continue

        for video_folder in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            video_path = os.path.join(category_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            for frame_file in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_file)
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(category)

    if not images:
        print("No images found. Please check the dataset structure.")
        return None, None, None

    images = np.array(images).reshape(-1, 48, 48, 1) / 255.0
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels, num_classes=len(CRIME_CLASSES))
    
    return images, labels, encoder

def train_model(data_path):
    """
    Train the crime detection model.
    """
    # Load data
    X, y, encoder = load_and_preprocess_data(data_path)
    if X is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile model
    model = create_crime_detection_model(num_classes=len(CRIME_CLASSES))
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Use a small number of epochs for a quick training session
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Generate classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))

    # Save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'crime_detection_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # You need to download the UCF-Crime dataset and extract it.
    # The dataset should be structured with subdirectories for each crime category.
    # For example: data/UCF-Crime/Abuse, data/UCF-Crime/Arrest, etc.
    # Update the path below to point to your dataset directory.
    # --- IMPORTANT ---
    dataset_path = '/Users/mukeshmishra/Downloads/surveillance-system/data/UCF-Crime'
    
    train_model(dataset_path)
