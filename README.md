# Cat_dog_classification
This repo takes 200 data of cats and dogs and takes a base model VGG16 and trains the fully connected layer on 200 images and do the model evaluation.
Here's a detailed README file for your project to upload on GitHub:

---

# Cat and Dog Image Classification using VGG16

## Overview

This project showcases a deep learning model built to classify images of cats and dogs using the VGG16 architecture. The model is trained using data augmentation techniques to improve robustness and accuracy. This repository contains the code and instructions to replicate the results.

## Features

- **Data Augmentation**: Enhances the dataset with random transformations.
- **Model Architecture**: Utilizes VGG16 for robust feature extraction.
- **Training and Evaluation**: Monitors performance with training and validation splits.
- **Single Image Prediction**: Classifies individual images as either a cat or a dog.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pillow
- Matplotlib

## Usage

### 1. Data Preparation

Extract the dataset:
```python
import zipfile

with zipfile.ZipFile("dc100.zip", 'r') as zip_ref:
    zip_ref.extractall()
```

### 2. Data Augmentation

Generate augmented images:
```python
data_aug('dc100', 'datagen', num_image=5)
```

### 3. Train the Model

Train the model with the augmented dataset:
```python
history = model.fit(X, Y, batch_size=24, epochs=10, validation_split=0.2, verbose=0, callbacks=[PrintLoss()])
```

### 4. Evaluate the Model

Plot training and validation accuracy:
```python
plt.plot(history.history['accuracy'], c='r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], c='b', label='Validation Accuracy')
plt.legend()
plt.title('Model accuracy')
plt.show()
```

Plot training and validation loss:
```python
plt.plot(history.history['loss'], c='r', label='Training Loss')
plt.plot(history.history['val_loss'], c='b', label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
```

### 5. Predict a Single Image

Classify an image as a cat or a dog:
```python
img_pred('5.jpg')
```

## Code Explanation

### Data Augmentation

This function generates augmented images:
```python
def data_aug(image_path, new_path, num_image=5):
    # Implementation here
```

### Reading Images

This function reads and processes images from the directory:
```python
def read_images(image_path):
    # Implementation here
```

### Model Definition

Defines and compiles the VGG16-based model:
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=((128, 128, 3)))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

### Training the Model

Custom callback to print loss:
```python
class PrintLoss(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1} loss: {logs['loss']}")
```

### Predicting Single Image

Function to predict and display the result for a single image:
```python
def img_pred(img_path):
    # Implementation here
```

## Contributing

Feel free to fork this repository, open issues, and submit pull requests.


This README file provides a clear and concise explanation of your project, its usage, and how others can contribute to it.
