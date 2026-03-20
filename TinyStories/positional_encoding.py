import keras 
import tensorflow as tf
import numpy as np

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model: int, seq_len: int, dtype=np.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        
        assert d_model%2==0
        
        pos=np.arange(seq_len)[:, np.newaxis]
        i=np.arange(d_model)[np.newaxis, :]
        
        pos_emb=np.zeros((1, seq_len, d_model))
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        
        pos_emb[0, :, 0::2]=np.sin(angle_rads[:, 0::2])
        pos_emb[0, :, 1::2]=np.cos(angle_rads[:, 1::2])
        
        self.pos_encoding=tf.constant(pos_emb.astype(dtype=dtype))
        self.supports_masking=True
    
    def call(self,X):
        batch_max_len=tf.shape(X)[1]
        return X+self.pos_encoding[:, :batch_max_len, :]   

 