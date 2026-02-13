import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_dim=1000),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
