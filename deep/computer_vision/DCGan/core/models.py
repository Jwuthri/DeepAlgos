# -*- coding: utf-8 -*-
"""
@author: JulienWuthrich
"""
import math
import glob

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf


class DCGan(object):
    """Deep Convolutional Generative Adversarial Networks."""

    def __init__(self, data_paths, batch_size=32, width=28, height=28, channels=3, beta1=.4, z_dim=100, alpha=.01, learning_rate=.005, mode='RGB', n_images=5):
        """Init.

        :param width: (int) input image width
        :param height: (int) input image height
        :param channels: (int) number of channels
        :param beta1: (float) exponential decay rate for the 1st moment in the optimizer
        :param z_dim: (int) z dimension
        :param alpha: (float) factor for the leaky relu
        :param learning_rate: (float) factor to control how much we are adjusting weights due to the loss
        :param mode: (str) how to read picture ('RGB', 'L')
        :param n_images: (int) number of images to display
        :param data_paths: (list) all images path
        :param batch_size: (int) batch size
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.beta1 = beta1
        self.z_dim = z_dim
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.mode = mode
        self.n_images = n_images
        self.data_paths = glob.glob(data_paths + "*")
        print("there are ", len(self. data_paths), "files")
        self.batch_size = batch_size
        self.out_shape = 4 * 4 * 256
        self.in_shape = 1 * 1 * 64

    def model_inputs(self):
        """Create model inputs.

        :return: (tuple) tensor of real input images, tensor of z data, learning rate
        """
        # placeholder is simply a variable that we will assign data to at a later date
        input_real = tf.placeholder(dtype=tf.float32, shape=[None, self.width, self.height, self.channels], name="input_real")
        input_z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name="input_z")
        learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")

        return input_real, input_z, learning_rate

    def discriminator(self, images, reuse=False):
        """Model to check if the images is a real one or a fake.
        logits: vector of raw (non-normalized) predictions that a classification model generates
        out: prediction of the model

        :param images: (tensor) images
        :param reuse: (boolean) reuse the weights or not
        :return: (tuple) tensor out of the discriminator, tensor logits of the discriminator
        """
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
            # Block 1 featuring
            conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
            relu1 = tf.maximum(conv1 * self.alpha, conv1)

            # Block 2 featuring
            conv2 = tf.layers.conv2d(inputs=relu1, filters=128, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            bn2 = tf.layers.batch_normalization(inputs=conv2, training=True)
            relu2 = tf.maximum(bn2 * self.alpha, bn2)
            drop2 = tf.nn.dropout(x=relu2, keep_prob=.8)

            # Block 3 featuring
            conv3 = tf.layers.conv2d(inputs=drop2, filters=256, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
            bn3 = tf.layers.batch_normalization(inputs=conv3, training=True)
            relu3 = tf.maximum(bn3 * self.alpha, bn3)
            drop3 = tf.nn.dropout(x=relu3, keep_prob=.8)

            # Last Block prediction
            shape = (-1, self.out_shape)
            flat = tf.reshape(tensor=drop3, shape=shape)
            logits = tf.layers.dense(inputs=flat, units=1)  # units: dimensionality of the output space.
            out = tf.sigmoid(logits)

            return logits, out

    def generator(self, input_z, out_channel_dim, is_train=True):
        """The goal of the generator is to generate passable hand-written digits, to lie without being caught.

        :param input_z: (tf.placeholder) input
        :param is_train: (boolean) use generator for the training
        :param out_channel_dim: (int) number of channel in output (data.shape[3])
        :return: (tensor) output of the generator
        """
        reuse = not is_train
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name_or_scope="generator", reuse=reuse):
            # Block 1 featuring
            dense1 = tf.layers.dense(inputs=input_z, units=self.out_shape)
            dense1 = tf.reshape(tensor=dense1, shape=(-1, 4, 4, 512))
            bn1 = tf.layers.batch_normalization(inputs=dense1, training=is_train)
            relu1 = tf.maximum(bn1 * self.alpha, bn1)

            # Block 2 featuring
            # conv2d_transpose: transform something that has the shape of the output of some convolution to something that has the shape of its input
            conv2_transpose = tf.layers.conv2d_transpose(inputs=relu1, filters=256, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
            bn2 = tf.layers.batch_normalization(inputs=conv2_transpose, training=is_train)
            relu2 = tf.maximum(bn2 * self.alpha, bn2)
            drop2 = tf.nn.dropout(x=relu2, keep_prob=.8)

            # Block 3 featuring
            conv3_transpose = tf.layers.conv2d_transpose(inputs=drop2, filters=128, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
            bn3 = tf.layers.batch_normalization(inputs=conv3_transpose, training=is_train)
            relu3 = tf.maximum(bn3 * self.alpha, bn3)
            drop3 = tf.nn.dropout(x=relu3, keep_prob=.8)

            # Last Block prediction
            logits = tf.layers.conv2d_transpose(inputs=drop3, filters=out_channel_dim, kernel_size=5, strides=1, padding="same")
            out = tf.tanh(logits)

            return out

    @staticmethod
    def loss_function(z, x):
        """Measures the probability error in discrete classification tasks
        in which each class is independent and not mutually exclusive.
        For instance, one could perform multilabels classification where
        a picture can contain both an elephant and a dog at the same time.

        :param z: labels
        :param x: logits
        :return: z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        """
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=z)

    def compute_loss(self, input_real, input_z, out_channel_dim):
        """Loss function for the discriminator and the generator.

        :param input_real: (tf.placeholder) real picture from dataset
        :param input_z: (tf.placeholder) inout z
        :param out_channel_dim: (int) number of channel in output (data.shape[3])
        :return: (tuple) discriminator loss, generator loss
        """
        gen = self.generator(input_z=input_z, out_channel_dim=out_channel_dim)
        disc_out_real, disc_logits_real = self.discriminator(images=input_real, reuse=False)
        disc_out_fake, disc_logits_fake = self.discriminator(images=gen, reuse=True)

        # measure the error
        # here wer use ones_like, cause the generated images, should be detect as real
        real_loss = self.loss_function(z=disc_out_real, x=tf.ones_like(tensor=disc_logits_real) * self.alpha)
        # here wer use zeros_like, cause the generated images, should be detect as fake
        fake_loss = self.loss_function(z=disc_out_fake, x=tf.zeros_like(tensor=disc_logits_fake) * self.alpha)
        # here wer use ones_like, cause this is real images
        gen_loss = self.loss_function(z=disc_out_fake, x=tf.ones_like(tensor=disc_logits_fake) * self.alpha)

        # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor
        disc_loss_real = tf.reduce_mean(real_loss)
        disc_loss_fake = tf.reduce_mean(fake_loss)
        gen_loss = tf.reduce_mean(gen_loss)

        return gen_loss, sum([disc_loss_real, disc_loss_fake])

    def optimization(self, disc_loss, gen_loss):
        """Optimize the loss.

        :param disc_loss: (tensor) discriminator loss
        :param gen_loss:  (tensor) generator loss
        :return: (tuple) discriminator training operation, generator training operation
        """
        tf_vars = tf.trainable_variables()
        disc_var = [var for var in tf_vars if var.name.startswith('discriminator')]
        gen_var = [var for var in tf_vars if var.name.startswith('generator')]

        operations = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)
        # control_dependencies: list of Operation or Tensor objects which must be executed or computed
        # before running the operations defined in the context
        with tf.control_dependencies(control_inputs=operations):
            disc_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1)
            disc_opt = disc_opt.minimize(loss=disc_loss, var_list=disc_var)
            gen_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1)
            gen_opt = gen_opt.minimize(loss=gen_loss, var_list=gen_var)

        return disc_opt, gen_opt

    def show_generator_output(self, sess, input_z):
        """Show example output for the generator

        :param sess: (tf.session) tensorFlow session
        :param input_z: (tensor) input z
        """
        cmap = None if self.mode == 'RGB' else 'gray'
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[self.n_images, z_dim])

        samples = sess.run(
            self.generator(input_z=input_z, out_channel_dim=self.channels, is_train=False),
            feed_dict={input_z: example_z}
        )

        images_grid = plot_images(samples, self.mode)
        plt.imshow(images_grid, cmap=cmap)
        plt.show()

    def get_batches(self):
        """Generate batches.

        :return: (np.array) batches of data
        """
        max_value = 255
        current_index = 0

        while current_index + self.batch_size <= len(self.data_paths):
            data_batch = get_images(
                self.data_paths[current_index:current_index + self.batch_size],
                self.width, self.height, self.mode
            )
            current_index += self.batch_size

            yield data_batch / max_value - 0.5

    @staticmethod
    def show_progression(epoch, epochs, steps, loss_disc, loss_gen):
        """Show progression of the gan.

        :param epoch: (int) current epoch
        :param epochs: (int) max epochs
        :param steps: (int) which batch step
        :param loss_disc: (float) disc loss
        :param loss_gen: (float) gen loss
        """
        print(
            "Epoch {}/{}... \n".format(epoch + 1, epochs),
            "Batch {}... \n".format(steps),
            "Discriminator loss {}... \n".format(loss_disc),
            "Generator loss {}... \n".format(loss_gen)
        )

    def train(self, epochs):
        """Train the DCGan.

        :param epochs: (int) number of epochs
        :return: (tuple) list of discriminator loss, list of generator loss
        """
        input_real, input_z, lr = self.model_inputs()
        disc_loss, gen_loss = self.compute_loss(input_real=input_real, input_z=input_z, out_channel_dim=self.channels)
        disc_opt, gen_opt = self.optimization(disc_loss=disc_loss, gen_loss=gen_loss)
        ldisc_loss, lgen_loss = list(), list()

        with tf.Session() as sess:
            sess.run(fetches=tf.global_variables_initializer())
            for epoch in range(epochs):
                steps = 0
                for batch in self.get_batches():
                    steps += 1
                    batch *= 2
                    rdm_batch = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.z_dim))
                    # feed_dict = {input_real: batch, input_z: rdm_batch, lr: self.learning_rate}
                    _ = sess.run(fetches=disc_opt, feed_dict={input_real: batch, input_z: rdm_batch, lr: self.learning_rate})
                    _ = sess.run(fetches=gen_opt, feed_dict={input_real: batch, input_z: rdm_batch, lr: self.learning_rate})

                    loss_disc = disc_loss.eval({input_real: batch, input_z: rdm_batch})
                    loss_gen = gen_loss.eval({input_z: rdm_batch})

                    ldisc_loss.append(loss_disc)
                    lgen_loss.append(loss_gen)

                    # show the gan progression
                    if steps % 20 == 0:
                        self.show_progression(epoch, epochs, steps, loss_disc, loss_gen)
                    # plot some generated images
                    if steps % 5 == 0:
                        self.show_generator_output(sess, input_z)

        return ldisc_loss, lgen_loss


def read_image(path, width, height, mode="RGB"):
    """Read image as a numpy array.

    :param path: (str) path of the image
    :param width: (int) new width size
    :param height: (int) new height size
    :param mode: (str) 'RGB' or 'L' image in color or gray
    :return: (np.array) image as np.array
    """
    image = Image.open(path)
    image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))


def get_images(paths, width, height, mode="RGB"):
    """Read all images and store them into a numpy array.

    :param paths: (list) image_path
    :param width: (int) new width size
    :param height: (int) new height size
    :param mode: (str) 'RGB' or 'L' image in color or gray
    :return: (np.array) array of images
    """
    images = [read_image(path, width, height, mode) for path in paths]
    images = np.array(images).astype(np.float32)
    # Be sure the images are in 4 dimensions
    if len(images.shape) < 4:
        images = images.reshape(images.shape + (1, ))

    return images


def plot_images(images, mode="RGB"):
    """Show the images in a square grid.

    :param images: (np.array) images to show in the grid
    :param mode: (str) 'RGB' or 'L' image in color or gray
    :return: (grid) grid square of images
    """
    # Maximal size of the grid square
    max_size = math.floor(np.sqrt(images.shape[0]))
    # Scale images
    images_scaled = ((images - images.min()) * 255) / (images.max() - images.min())
    images_scaled = images_scaled.astype(np.uint8)
    # Arrange images in the grid
    a = images_scaled[:max_size * max_size]
    new_shape = (max_size, max_size, *images_scaled.shape[:3])
    images_squared = np.reshape(a, new_shape)
    # Remove single-dimensional entries from the shape of an array
    if mode == 'L':
        images_squared = np.squeeze(images_squared, 4)
    # Combine image into the grid
    size = (images[1] * max_size, images.shape[2] * max_size)
    grid = Image.new(mode, size)
    for col_i, col_images in enumerate(images_squared):
        for image_i, image in enumerate(col_images):
            img = Image.fromarray(image, mode)
            box = (col_i * images.shape[1], image_i * images.shape[2])
            grid.paste(img, box)

    return grid


if __name__ == '__main__':
    gan = DCGan(data_paths="../../../../data/raw/dcgan/")
    with tf.Graph().as_default():
        disc_loss, gen_loss = gan.train(epochs=100)
