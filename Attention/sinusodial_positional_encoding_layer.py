import keras 
import tensorflow as tf 
import numpy as np

class SinusodialPositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_len, d_model, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        
        assert d_model%2==0 # Should be even
        
        p=np.arange(seq_len) # Creates position indices from 0 to seq_len-1
        i=np.arange(d_model//2) 
        
        pos_emb=np.empty((1, seq_len, d_model)) # Creates an empty array of shape (1, seq_len, d_model)
        
        pos_emb[0,:,::2]=np.sin(p/10000**(2*i/d_model))
        pos_emb[0,:,1::2]=np.cos(p/10000**(2*i/d_model))
        
        self.pos_encoding=tf.constant(pos_emb.astype(self.dtype)) # Converts numpy array to tensor
        self.supports_masking=True
        
    def call(self,X):
        batch_max_len=tf.shape(X)[1] # Gets the seq_len of the input tensor
        return X+self.pos_encoding[:, :batch_max_len, :] # Passes the seq_len of the input tensor to the pos_encoding layer and adds it to the input tensor
