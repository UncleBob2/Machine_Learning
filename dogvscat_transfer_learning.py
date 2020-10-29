import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from tensorflow.keras import layers

(train_examples, validation_examples), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    with_info=True,
    as_supervised=True,
)


def format_image(image, label):
    # `hub` image modules exepct their data normalized to the [0,1] range.
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


num_examples = info.splits["train"].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224

train_batches = (
    train_examples.cache()
    .shuffle(num_examples // 4)
    .map(format_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
validation_batches = (
    validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))


feature_extractor.trainable = False

model = tf.keras.Sequential([feature_extractor, layers.Dense(2)])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

EPOCHS = 3
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)
