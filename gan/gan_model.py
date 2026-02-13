import tensorflow as tf
from generator import build_generator
from discriminator import build_discriminator

latent_dim = 100

generator = build_generator(latent_dim)
discriminator = build_discriminator()
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
fake_news = generator(gan_input)
gan_output = discriminator(fake_news)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer="adam")
