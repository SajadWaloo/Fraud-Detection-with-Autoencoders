import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv3DTranspose, Conv3D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define the Generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 8 * 8 * 4, input_dim=latent_dim))
    model.add(Reshape((8, 8, 4, 128)))
    model.add(Conv3DTranspose(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Conv3D(1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='sigmoid'))
    return model

# Define the Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', input_shape=(8, 8, 4, 1)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Combine Generator and Discriminator into a GAN model
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# Load and preprocess the Mobilenet40 dataset
def load_dataset(dataset_path):
    dataset = []
    # Iterate through the dataset files
    for filename in os.listdir(dataset_path):
        # Load and preprocess each voxel grid file
        filepath = os.path.join(dataset_path, filename)
        voxel_grid = preprocess_voxel_grid(filepath)
        dataset.append(voxel_grid)
    
    dataset = np.array(dataset)
    return dataset

# Preprocess a single voxel grid file
def preprocess_voxel_grid(filepath):
    # Implement your voxel grid preprocessing logic here
    # ...

    return voxel_grid

# Define training parameters
latent_dim = 100
epochs = 100
batch_size = 32

# Build the Generator and Discriminator models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Build the GAN model
gan = build_gan(generator, discriminator)

# Compile the Discriminator and GAN models
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Set the path to the Mobilenet40 dataset
dataset_path = "/Users/mac/Desktop/Projects/Datasets/archiv"

# Load and preprocess the Mobilenet40 dataset
dataset = load_dataset(dataset_path)

print("Dataset loaded successfully.")

# Training loop
for epoch in range(epochs):
    for batch in range(len(dataset) // batch_size):
        # Generate random noise as input to the Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate 3D object samples using the Generator
        generated_objects = generator.predict(noise)

        # Select a batch of real 3D objects from the dataset
        real_objects = dataset[batch * batch_size: (batch + 1) * batch_size]

        # Train the Discriminator
        discriminator_loss_real = discriminator.train_on_batch(real_objects, np.ones(batch_size))
        discriminator_loss_generated = discriminator.train_on_batch(generated_objects, np.zeros(batch_size))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

        # Train the Generator (via the GAN model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan_loss = gan.train_on_batch(noise, np.ones(batch_size))

    # Print training progress
    print(f"Epoch {epoch+1}/{epochs} - Discriminator Loss: {discriminator_loss[0]}, Discriminator Accuracy: {discriminator_loss[1]}, GAN Loss: {gan_loss}")

    # Generate some 3D object samples after every 10 epochs
    if epoch % 10 == 0:
        num_samples = 1
        generated_samples = generator.predict(np.random.normal(0, 1, (num_samples, latent_dim)))

        # Display the generated 3D object samples
        for i in range(num_samples):
            sample = generated_samples[i]
            plt.figure()
            plt.imshow(sample.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title(f"Generated Sample {i+1}")
            plt.show()

# Generate some 3D object samples after training
num_samples = 10
generated_samples = generator.predict(np.random.normal(0, 1, (num_samples, latent_dim)))

# Display the generated 3D object samples
for i in range(num_samples):
    sample = generated_samples[i]
    plt.figure()
    plt.imshow(sample.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Generated Sample {i+1}")
    plt.show()

print("Model training completed.")

# Display or save the generated 3D object samples
for i in range(num_samples):
    sample = generated_samples[i]
    # Display or save the generated 3D object sample
    # ...

# Print additional results or metrics
# ...
