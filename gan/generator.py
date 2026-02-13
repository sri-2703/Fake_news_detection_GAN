import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_dim=latent_dim),
        layers.Dense(500, activation="relu"),
        layers.Dense(1000, activation="linear")
    ])
    return model
