import keras
import tensorflow as tf
import numpy as np

@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model: int, seq_len: int, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        self.d_model = int(d_model)
        self.seq_len = int(seq_len)

        pos = np.arange(self.seq_len)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates

        pos_emb = np.zeros((1, self.seq_len, self.d_model), dtype=np.float32)
        pos_emb[0, :, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_emb[0, :, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(pos_emb, dtype=self.dtype)
        self.supports_masking = True

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_len": self.seq_len,
        })
        return config