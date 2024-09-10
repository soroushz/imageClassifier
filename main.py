import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import random

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

model_path = "cifar10_cnn_model.h5"
if os.path.exists(model_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from disk.")
else:
    # Build and train the model if it doesn't exist
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    # Save the model
    model.save(model_path)
    print("Model saved to disk.")


def classify_image(image):
    img_array = tf.expand_dims(image, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class]


# Display the image with its predicted class with improved visual quality
def show_image_with_prediction(image, true_label):
    predicted_label = classify_image(image)

    plt.figure(figsize=(2, 2))  # Reduce figure size to avoid pixelation
    plt.imshow(image, interpolation="nearest")  # Disable interpolation for sharper display
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis("off")
    plt.show()


# Pick a random image from the test set and classify it
random_index = random.randint(0, len(test_images) - 1)
random_image = test_images[random_index]
true_label = class_names[test_labels[random_index][0]]  # Get the true label from test_labels

class_pred = classify_image(random_image)
show_image_with_prediction(random_image, true_label)

