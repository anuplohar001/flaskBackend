import numpy as np
import cv2
import tensorflow as tf
import os


model = tf.keras.models.load_model("models/skin.h5")


label_map = {0: "Eczema", 1: "Psoriasis", 2: "Acne", 3: "Healthy"}


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  
    img = img.astype('float32') / 255.0  
    return np.expand_dims(img, axis=0)  


test_cases = {
    "test_images/eczema_1.jpg": "Eczema",
    "test_images/blurry.jpg": "Invalid",
}


for path, expected in test_cases.items():
    try:
        img = preprocess_image(path)
        pred = model.predict(img)
        predicted_label = label_map[np.argmax(pred)]

        if expected == "Invalid":
            print(f"{path}: ❌ Expected Error but got {predicted_label}")
        elif predicted_label == expected:
            print(f"{path}: ✅ Pass (Predicted: {predicted_label})")
        else:
            print(f"{path}: ❌ Fail (Predicted: {predicted_label}, Expected: {expected})")
    except Exception as e:
        if expected == "Invalid":
            print(f"{path}: ✅ Pass (Caught expected error: {str(e)})")
        else:
            print(f"{path}: ❌ Fail (Unexpected error: {str(e)})")
