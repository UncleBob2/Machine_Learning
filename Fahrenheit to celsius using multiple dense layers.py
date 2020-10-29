# mkvirtualenv tensor2
# ls $WORKON_HOME
# workon tensor2
# rmvirtualenv
# deactivate


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging


logger = tf.get_logger()
logger.setLevel(logging.ERROR)
celsius_feature = np.array(
    [-40, -10, 0, 8, 15, 22, 38], dtype=float
)  # features are input
fahrenheit_labels = np.array(
    [-40, 14, 32, 46, 59, 72, 100], dtype=float
)  # labels are output

for i, c in enumerate(celsius_feature):
    print(
        "Training examples: {} {} degrees Celsius (features) = {} degrees Fahrenheit (labels)".format(
            i, c, fahrenheit_labels[i]
        )
    )


"""
layer0 = tf.keras.layers.Dense(
    units=1, input_shape=[1]
)  # define single dense layer with celsius_future as input, units define how many internal variable this dense layer will have

model = tf.keras.Sequential([layer0])  # using single layer0
model.compile(
    loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1)
)  # define loss using mean_square_error and optimizers.Adam is the learning rate range from 0.001 to 0.1

# begining training by calling the model.fit with i/p celsius_feature, o/p labels fahrenheits, 1 epoch is a full example of what we
# provide of sample set. Therefore 500 epochs equals to 500x7 =3,500 training examples
history = model.fit(celsius_feature, fahrenheit_labels, epochs=500, verbose=False)  #
"""
layer0 = tf.keras.layers.Dense(units=4, input_shape=[1])
layer1 = tf.keras.layers.Dense(units=4)
layer2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([layer0, layer1, layer2])
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_feature, fahrenheit_labels, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))

print(
    "These are the layer variables of weights and bias layer0 variables: {}".format(
        layer0.get_weights()
    )
)
print("These are the layer1 variables: {}".format(layer1.get_weights()))
print("These are the layer2 variables: {}".format(layer2.get_weights()))


print("Predicted value for 100C/212F is ", model.predict([100.0]))


plt.title("Loss Magnitude vs. Epoch Number")
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])

plt.show()