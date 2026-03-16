"""
Custom Keras layers for the Flash Crash prediction models.
Import this module before loading any model that uses custom layers.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """Bahdanau-style additive attention over timesteps."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_W",
            shape=(int(input_shape[-1]), int(input_shape[-1])),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_b",
            shape=(int(input_shape[-1]),),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(int(input_shape[-1]),),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[[2], [0]]), axis=1
        )  # (batch, timesteps)
        context = tf.reduce_sum(
            x * tf.expand_dims(attention_weights, -1), axis=1
        )  # (batch, features)
        return context

    def get_config(self):
        return super().get_config()
