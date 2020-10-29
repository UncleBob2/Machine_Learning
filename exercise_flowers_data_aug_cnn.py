import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), "flower_photos")

classes = ["roses", "daisy", "dandelion", "sunflowers", "tulips"]

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + "/*.jpg")
    print("{}: {} Images".format(cl, len(images)))
    train, val = images[: round(len(images) * 0.8)], images[round(len(images) * 0.8) :]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, "train", cl)):
            os.makedirs(os.path.join(base_dir, "train", cl))
        shutil.move(t, os.path.join(base_dir, "train", cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, "val", cl)):
            os.makedirs(os.path.join(base_dir, "val", cl))
        shutil.move(v, os.path.join(base_dir, "val", cl))

round(len(images) * 0.8)

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

batch_size = 100
IMG_SHAPE = 150

image_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=45)


train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5,
)


train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1.0 / 255)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            input_shape=(IMG_SHAPE, IMG_SHAPE, 3),
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

epochs = 80

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

"""
In this lesson we learned how Convolutional Neural Networks work with color images and saw various techniques that we can use to avoid overfitting . The main key points of this lesson are:

CNNs with RGB Images of Different Sizes:

    Resizing: When working with images of different sizes, you must resize all the images to the same size so that they can be fed into a CNN.

    Color Images: Computers interpret color images as 3D arrays.

    RGB Image: Color image composed of 3 color channels: Red, Green, and Blue.

    Convolutions: When working with RGB images we convolve each color channel with its own convolutional filter. Convolutions on each color channel are performed in the same way as with grayscale images, i.e. by performing element-wise multiplication of the convolutional filter (kernel) and a section of the input array. The result of each convolution is added up together with a bias value to get the convoluted output.

    Max Pooling: When working with RGB images we perform max pooling on each color channel using the same window size and stride. Max pooling on each color channel is performed in the same way as with grayscale images, i.e. by selecting the max value in each window.

    Validation Set: We use a validation set to check how the model is doing during the training phase. Validation sets can be used to perform Early Stopping to prevent overfitting and can also be used to help us compare different models and choose the best one.

Methods to Prevent Overfitting:

    Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.

    Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.

    Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.

You also created and trained a Convolutional Neural Network to classify images of Dogs and Cats with and without Image Augmentation and Dropout. You were able to see that Image Augmentation and Dropout greatly reduces overfitting and improves accuracy. As an exercise, you were able to apply everything you learned in this lesson to create your own CNN to classify images of flowers. """
