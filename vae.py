import tensorflow as tf
from tensorflow.keras import Input, layers, models
import numpy as np


class VAE:
    def __init__(self, input_shape: tuple[int] = (128, 128, 3), latent_dim: int = 2):

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

        # Model layer; initialized by build_model
        self.outputs = None

    def _build_encoder(self):

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
        self.flatten = layers.Flatten(name="Encoder-Flatten")(self.conv2d_2)

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

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(
            shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0.0, stddev=1.0
        )
        return z_mean + tf.math.exp(z_log_sigma) * epsilon

    def _build_decoder(self):

        # Input layer
        self.latent_inputs = Input(
            shape=(self.latent_dim,), name="Input-from-Z-Sampling"
        )

        # Hidden layers: Reshaping
        self.dense = layers.Dense(
            units=16 * 16 * 32, activation="relu", name="Decoder-Dense"
        )(
            self.latent_inputs
        )  # HARD CODED, NEEDS TINKERING
        self.reshape = layers.Reshape(
            target_shape=(16, 16, 32), name="Decoder-Reshape"
        )(self.dense)

        # Hidden layers: Upsampling
        self.conv2dtranspose_1 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name="Decoder-Conv2DTranspose-1",
        )(self.reshape)
        self.conv2dtranspose_2 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name="Decoder-Conv2DTranspose-2",
        )(self.conv2dtranspose_1)
        self.conv2dtranspose_3 = layers.Conv2DTranspose(
            filters=3,
            kernel_size=3,
            strides=1,
            padding="same",
            name="Decoder-Conv2DTranspose-3",
        )(
            self.conv2dtranspose_2
        )  # no activation!

        # Create decoder model
        self.decoder = models.Model(
            self.latent_inputs, self.conv2dtranspose_3, name="Decoder"
        )

    def build_model(self):
        self._build_encoder()
        self._build_decoder()
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.model = models.Model(
            inputs=self.inputs, outputs=self.outputs, name="VAE-Model"
        )

    def loss_function(self, inputs, outputs):
        # def loss_function(self):
        # r_loss = np.product(self.input_shape) * tf.keras.losses.mse(
        #     self.inputs, self.outputs
        # )
        r_loss = np.product(self.input_shape) * tf.keras.losses.mse(inputs, outputs)
        kl_loss = -0.5 * tf.math.reduce_sum(
            1
            + self.z_log_sigma
            - tf.math.square(self.z_mean)
            - tf.math.exp(self.z_log_sigma),
            axis=1,
        )
        vae_loss = tf.math.reduce_mean(r_loss + kl_loss)

        return vae_loss
