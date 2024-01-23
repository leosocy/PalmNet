import cv2 as cv
import keras
import numpy as np
import tensorflow as tf


class GaussianLayer(keras.layers.Layer):
    def __init__(self, kernel_size=3, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        kernel = self._gaussian_kernel(self.kernel_size, self.sigma)
        kernel = np.array(kernel)
        self.kernel = self.add_weight(
            "gaussian_kernel",
            shape=(self.kernel_size, self.kernel_size, 1, 1),
            initializer=tf.constant_initializer(kernel),
            trainable=False,
        )

    def call(self, inputs, *args, **kwargs):
        return tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME")

    def _gaussian_kernel(self, size, sigma):
        ax = tf.range(-size // 2 + 1.0, size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / tf.reduce_sum(kernel)


class LaplaceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32, shape=(3, 3, 1, 1))

    def call(self, inputs, *args, **kwargs):
        laplace = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.add(inputs, laplace)


class GaborConv2D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        activation=None,
        trainable=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self._trainable = trainable

    def build(self, input_shape):
        gabor_kernels = []
        for theta in np.linspace(0, np.pi, self.filters, endpoint=False):
            gabor_kernel = cv.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                sigma=self.kernel_size / np.pi,
                theta=theta,
                lambd=self.kernel_size / np.pi,
                gamma=1,
                psi=0,
            )
            gabor_kernel = np.expand_dims(gabor_kernel, axis=-1)
            gabor_kernel = tf.math.real(gabor_kernel)
            gabor_kernels.append(gabor_kernel)
        gabor_kernels = np.stack(gabor_kernels, axis=-1)

        self.gabor_kernels = self.add_weight(
            name="gabor_kernels",
            shape=gabor_kernels.shape,
            initializer=tf.constant_initializer(gabor_kernels),
            trainable=self._trainable,
        )

    def call(self, inputs, *args, **kwargs):
        inputs = tf.nn.conv2d(
            inputs,
            self.gabor_kernels,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper(),
        )
        if self.activation:
            inputs = self.activation(inputs)
        return inputs


class ArcLayer(keras.layers.Layer):
    """Custom layer for ArcFace.

    This layer is equivalent a dense layer except the weights are normalized.
    """

    def __init__(self, units, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=[input_shape[-1], self.units],
            dtype=tf.float32,
            initializer=keras.initializers.HeNormal(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel",
        )
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "kernel_regularizer": self.kernel_regularizer})
        return config
