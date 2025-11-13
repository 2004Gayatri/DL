!pip install tensorflow

---------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
--------------------------------

train_dir = r"C:\Users\sanap\Downloads\cifar-10-img1\cifar-10-img\train"
test_dir = r"C:\Users\sanap\Downloads\cifar-10-img1\cifar-10-img\test"
-------------------------------------------------------

train_data = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(32, 32),    # resize images to 32x32
    batch_size=32,
    label_mode='int'        # integer labels for sparse categorical loss
)

test_data = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode='int'
)
------------------------------------------------------------

# Normalize pixel values (0–255 → 0–1)
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_data = train_data.map(normalize_img) 
test_data = test_data.map(normalize_img)

# Flatten the image tensors for feedforward NN
train_data = train_data.map(lambda x, y: (tf.reshape(x, (tf.shape(x)[0], -1)), y))
test_data = test_data.map(lambda x, y: (tf.reshape(x, (tf.shape(x)[0], -1)), y))


--------------------------------------------------------------

# ---- Step c: Define Model Architecture ----
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(32*32*3,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
-------------------------------------------------------------
# ---- Step d: Compile & Train ----
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)
------------------------------------------------------------------
# ---- Step e: Evaluate ----
test_loss, test_acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
-------------------------------------------------------------
# ---- Step f: Plot Training & Validation ----
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
------------------------------------------------------------------
# ---- Step g: Predict Example ----
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Take one batch from test data
for images, labels in test_data.take(1):
    predictions = model.predict(images)
    index = 5  # choose any sample index

    # Convert from flattened shape to (32, 32, 3)
    image_np = images[index].numpy().reshape(32, 32, 3)

    plt.imshow(image_np)
    plt.axis('off')
    plt.show() 

    print("Predicted Class:", class_names[np.argmax(predictions[index])])
    print("Actual Class:", class_names[int(labels[index])])
--------------------------------------------------------------