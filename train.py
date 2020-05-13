"""
Created on Thu May 09 17:56:24 2020
@author: Kraken

Project: GAN for MNIST dataset using Keras API
"""
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.datasets import mnist

if not os.path.exists('model_evaluation'):
    os.mkdir('model_evaluation')


# Import Dataset
def import_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Train', x_train.shape, y_train.shape)
    print('Test', x_test.shape, y_test.shape)
    # plt.imshow(x_train[0], cmap='gray_r')

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    # scale pixel values in range [0, 1]
    x_train /= 255.0
    return x_train


# Generate Dataset
def sample_real_images(dataset, batch_size):
    index = np.random.randint(0, dataset.shape[0], batch_size)
    X = dataset[index]
    # real class labels
    y = np.ones((batch_size, 1))
    return X, y


def generate_latent_points(latent_dim, batch_size):
    gen_X = np.random.randn(latent_dim * batch_size)
    gen_X = gen_X.reshape((batch_size, latent_dim))
    return gen_X


def generate_fake_images(model, latent_dim, batch_size):
    x_input = generate_latent_points(latent_dim, batch_size)
    X = model.predict(x_input)
    # fake class labels
    y = np.zeros((batch_size, 1))
    return X, y


# Design Models
def create_generator(latent_dim=100):
    model = Sequential(name='Generator')
    # get 7 x 7 image
    model.add(Dense(units=7 * 7 * 128, use_bias=False,
                    input_shape=(latent_dim,),
                    name='input_layer'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))

    # get 14 x 14 image
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              use_bias=False,
                              name='Upsample_1'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # get 28 x 28 image
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4),
                              strides=(2, 2), padding='same',
                              use_bias=False,
                              name='Upsample_2'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=1, kernel_size=(7, 7),
                     activation='sigmoid',
                     padding='same',
                     name='output_layer'))
    # no compilation of generator model as it is
    # used as intermediate model in GAN_MODEL function
    return model


def create_discriminator():
    model = Sequential(name='Discriminator')
    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                     strides=(2, 2), padding='same',
                     input_shape=[28, 28, 1],
                     name='Conv_1'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))

    model.add(Conv2D(filters=128, kernel_size=(5, 5),
                     strides=(2, 2), padding='same',
                     name='Conv_2'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))

    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid',
                    name='output'))

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_gan_model(generator, discriminator):
    discriminator.trainable = False

    model = Sequential(name='GAN_Model')
    model.add(generator)
    model.add(discriminator)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')
    return model


def get_models(latent_dim):
    generator = create_generator(latent_dim=latent_dim)
    # generator.summary()
    # plot_model(generator, to_file='generator_architecture.png',
    #            show_shapes=True, show_layer_names=True)

    discriminator = create_discriminator()
    # discriminator.summary()
    # plot_model(discriminator, to_file='discriminator_architecture.png',
    #            show_shapes=True, show_layer_names=True)

    gan = create_gan_model(generator, discriminator)
    # gan.summary()
    # plot_model(gan, to_file='gan_architecture.png',
    #            show_shapes=True, show_layer_names=True)

    # # check sample generator output
    # noise = np.random.randn(100).reshape((1, 100)).astype('float32')
    # generated_image = generator(noise, training=False)
    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    # # check sample discriminator output
    # decision = discriminator(generated_image)
    # print(decision)
    return generator, discriminator, gan


def save_image(examples, epoch, n=5):
    fig = plt.figure(figsize=(5, 5))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0] * 255, cmap='gray_r')

    fig.tight_layout()
    plt.suptitle('Epoch : %03d' % (epoch))
    plt.savefig('model_evaluation/image_%03d.png' % (epoch))
    plt.close()


def model_evaluation(epoch, generator, discriminator,
                     dataset, latent_dim, n_samples=100):
    X_real, y_real = sample_real_images(dataset, n_samples)
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)

    X_fake, y_fake = generate_fake_images(generator, latent_dim,
                                          n_samples)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)
    print('->Accuracy Real:{:5.2f}%, Fake:{:5.2f}%'
          .format(acc_real * 100, acc_fake * 100))
    save_image(X_fake, epoch)
    generator.save('model_evaluation/g_model_%03d.h5' % (epoch))
    discriminator.save('model_evaluation/d_model_%03d.h5' % (epoch))


def create_animation():
    anim_file = 'results2.gif'
    img_dir = 'model_evaluation'
    images = []
    for file_name in sorted(os.listdir(img_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(img_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(anim_file, images, fps=2)


def train(generator, discriminator, gan, dataset,
          latent_dim, n_epochs=100, n_batch=256):
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for epoch in range(1, n_epochs + 1):
        start = time.time()
        for i in range(1, batch_per_epoch + 1):
            X_real, y_real = sample_real_images(dataset, half_batch)
            X_fake, y_fake = generate_fake_images(generator,
                                                  latent_dim,
                                                  half_batch)
            # combine real and fake images
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # train discriminator
            d_loss, _ = discriminator.train_on_batch(X, y)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            # train generator
            g_loss = gan.train_on_batch(X_gan, y_gan)
        end = time.time()
        print('Epoch: {:3d}/{:3d} | d_loss: {:.3f} | g_loss: {:.3f} | '
              'Time: {:5.2f} min'
              .format(epoch, n_epochs, d_loss, g_loss, (end - start) / 60))
        if epoch % 10 == 0:
            model_evaluation(epoch, generator, discriminator,
                             dataset, latent_dim)


if __name__ == "__main__":
    x_train = import_mnist_dataset()
    latent_dim = 100
    generator, discriminator, gan = get_models(latent_dim)

    train(generator, discriminator, gan, x_train, latent_dim, n_epochs=100)
    create_animation()
