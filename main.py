import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
import numpy as np
from PIL import Image

# Set the path to the dataset
data_path = 'C:\\GANs\\images1'

# Define the parameters
input_shape = (128, 128, 3)  # Assuming the images are RGB and 128x128 pixels
latent_dim = 100  # Dimensionality of the random noise vector
batch_size = 32
epochs = 1000

# Load the dataset
def load_data(data_path):
    images = []
    for filename in os.listdir(data_path):
        img = Image.open(os.path.join(data_path, filename))
        img = np.array(img.resize(input_shape[:2])) / 255.0  # Normalize the images to the range [0, 1]
        images.append(img)
    return np.array(images)

# Create the generator model
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(128 * 16 * 16, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Create the discriminator model
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Define the loss functions
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

# Create the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Create the generator and discriminator optimizers
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

# Generate a fixed noise vector
fixed_noise = tf.random.normal([batch_size, latent_dim])

# Define the training step
@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fixed_noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Load the dataset
dataset = load_data(data_path)
dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(10000).batch(batch_size)

# Train the GAN
for epoch in range(epochs):
    for image_batch in dataset:
        train_step(image_batch)

    # Save generated images every 100 epochs
    if (epoch + 1) % 100 == 0:
        generated_images = generator(fixed_noise, training=False)
        generated_images = tf.cast((generated_images * 255), tf.uint8)
        for i, image in enumerate(generated_images):
            Image.fromarray(image.numpy()).save(f'generated_image_epoch_{epoch + 1}_{i}.png')


