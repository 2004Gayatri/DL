import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
---------------------------------
# path of dataset
train_dir = r"C:\Users\sanap\Downloads\mnist-jpg\mnist-jpg\train"
test_dir = r"C:\Users\sanap\Downloads\mnist-jpg\mnist-jpg\test"

# load the dataset
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = (28,28),
    color_mode = 'grayscale', 
    batch_size = 32
)
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = (28,28),
    color_mode = 'grayscale',
    batch_size = 32 
)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y:(normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y:(normalization_layer(x), y))


-----------------------------------------------------------
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),     # Flatten grayscale 28x28 image to 1D
    layers.Dense(128, activation='relu'),         # Hidden layer 1
    layers.Dense(64, activation='relu'),          # Hidden layer 2
    layers.Dense(10, activation='softmax')        # Output layer (10 classes: digits 0–9)
]) 
----------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
) 
--------------------------------------------------------------

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print("\nTest accuracy:", test_acc)
print("Test loss:", test_loss)
--------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
------------------------------------------------------------------
# Get one batch of test images and labels
x_test, y_test = next(iter(test_ds))

# Make predictions
pred = model.predict(x_test)
pred_classes = np.argmax(pred, axis=1)

# Pick any image to test
i = 5  # change the index (0–31) to test another image

# Show the image with true and predicted labels
plt.imshow(x_test[i].numpy().reshape(28, 28), cmap='gray')
plt.title(f"True: {y_test[i].numpy()}, Predicted: {pred_classes[i]}")
plt.axis('off')
plt.show()

# Print classes
print("Predicted Class:", pred_classes[i]) 
print("Actual Class:", int(y_test[i].numpy()))
---------------------------------------------------------------