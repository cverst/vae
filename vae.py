import tensorflow as tf
from tensorflow.keras import Input, layers, models
import numpy as np


class VAE:
    """Convolutional Variational Autoencoder"""

    def __init__(
        self, input_shape: tuple[int] = (128, 128, 3), latent_dim: int = 2
    ) -> None:

        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Models; initialized later
        self.encoder = None
        self.decoder = None
        self.model = None

        # Encoder layers; initialized by _build_encoder
        self.inputs = None
        self.conv2d_1 = None
        self.conv2d_2 = None
        self.conv2d_3 = None
        self.flatten = None
        self.z_mean = None
        self.z_log_sigma = None
        self.z = None

        # Decoder layers; initialized by _build_decoder
        self.latent_inputs = None
        self.dense = None
        self.reshape = None
        self.conv2dtranspose_1 = None
        self.conv2dtranspose_2 = None
        self.conv2dtranspose_3 = None
        self.conv2dtranspose_4 = None

        # Model layer; initialized by build_model
        self.outputs = None

    def _build_encoder(self) -> None:
        """Build encoder part.

        The encoder uses three convolutional layers before entering the final
        bottleneck of size model.latent_dim.
        """

        # Input layer
        self.inputs = Input(shape=self.input_shape, name="Input")

        # Hidden layers
        self.conv2d_1 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation="relu",
            name="Encoder-Conv2D-1",
        )(self.inputs)
        self.conv2d_2 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation="relu",
            name="Encoder-Conv2D-2",
        )(self.conv2d_1)
        self.conv2d_3 = layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation="relu",
            name="Encoder-Conv2D-3",
        )(self.conv2d_2)
        self.flatten = layers.Flatten(name="Encoder-Flatten")(self.conv2d_3)

        # Latent space layer; no activations!
        self.z_mean = layers.Dense(units=self.latent_dim, name="Z-Mean")(self.flatten)
        self.z_log_sigma = layers.Dense(units=self.latent_dim, name="Z-Log-Sigma")(
            self.flatten
        )
        self.z = layers.Lambda(self._sampling, name="Z-Sampling-Layer")(
            [self.z_mean, self.z_log_sigma]
        )

        # Create encoder model
        self.encoder = models.Model(
            self.inputs, [self.z_mean, self.z_log_sigma, self.z], name="Encoder"
        )

    def _sampling(self, args: list) -> list:
        """Sampling of latent distributions.

        Args:
            args (list): Distribution parameters in the form of [z_mean, z_log_sigma].

        Returns:
            list: Sampled vector.
        """
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(
            shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0.0, stddev=1.0
        )
        return z_mean + tf.math.exp(z_log_sigma) * epsilon

    def _build_decoder(self) -> None:
        """Build decoder part."""

        # Input layer
        self.latent_inputs = Input(
            shape=(self.latent_dim,), name="Input-from-Z-Sampling"
        )

        # Hidden layers: reshaping
        target_shape = np.multiply(
            self.encoder.get_layer("Encoder-Conv2D-3").output_shape[1:], [1, 1, 0.5]
        ).astype(int)
        self.dense = layers.Dense(
            units=np.product(target_shape), activation="relu", name="Decoder-Dense"
        )(self.latent_inputs)
        self.reshape = layers.Reshape(
            target_shape=target_shape, name="Decoder-Reshape"
        )(self.dense)

        # Hidden layers: upsampling
        self.conv2dtranspose_1 = layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name="Decoder-Conv2DTranspose-1",
        )(self.reshape)
        self.conv2dtranspose_2 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name="Decoder-Conv2DTranspose-2",
        )(self.conv2dtranspose_1)
        self.conv2dtranspose_3 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name="Decoder-Conv2DTranspose-3",
        )(self.conv2dtranspose_2)
        self.conv2dtranspose_4 = layers.Conv2DTranspose(
            filters=self.input_shape[2],
            kernel_size=3,
            strides=1,
            padding="same",
            activation="tanh",
            name="Decoder-Conv2DTranspose-4",
        )(self.conv2dtranspose_3)

        # Create decoder model
        self.decoder = models.Model(
            self.latent_inputs, self.conv2dtranspose_4, name="Decoder"
        )

    def build_model(self) -> None:
        """Connect encoder and decoder to create full model."""

        self._build_encoder()
        self._build_decoder()
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.model = models.Model(
            inputs=self.inputs, outputs=self.outputs, name="Convolutional-VAE-Model"
        )

    def compile_model(self) -> None:
        """Compile model after creating loss function."""

        # Loss based on MSE and Kullback-Leibler divergence
        r_loss = np.product(self.input_shape) * tf.math.reduce_sum(
            tf.math.reduce_sum(tf.keras.losses.mse(self.inputs, self.outputs), axis=1),
            axis=1,
        )
        kl_loss = -0.5 * tf.math.reduce_sum(
            1
            + self.z_log_sigma
            - tf.math.square(self.z_mean)
            - tf.math.exp(self.z_log_sigma),
            axis=1,
        )
        vae_loss = tf.math.reduce_mean(r_loss + kl_loss)

        self.model.add_loss(vae_loss)
        self.model.compile(optimizer="adam")
