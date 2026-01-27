import tensorflow as tf 
import keras 
import numpy as np 
from multi_head import multi_head_attention

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__() # Initialises the parent Layer class for keras to know to track weights
        
        assert d_model%num_heads==0 # Crashes if condition is not true
        
        self.num_heads=num_heads
        self.d_model=d_model
        
        self.d_k=d_model//num_heads
        
        self.W_q=keras.layers.Dense(d_model, use_bias=False) # A dense layer automatically assigns random weights and makes them trainable
        self.W_k=keras.layers.Dense(d_model, use_bias=False)
        self.W_v=keras.layers.Dense(d_model, use_bias=False)
        self.W_o=keras.layers.Dense(d_model, use_bias=False)
    
    def call(self, X):
        self.Q=self.W_q(X)
        self.K=self.W_k(X)
        self.V=self.W_v(X)
        
        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)
        
        scores=tf.matmul(Q, K, transpose_b=True)/tf.math.sqrt(tf.cast(self.d_k,tf.float32))
        
        weights=tf.nn.softmax(scores, axis=-1)
        attention=tf.matmul(weights, V)
        
        attention=self.combine_heads(attention)
        
        return self.W_o(attention)
    
    def split_heads(self, X):
        batch_size=tf.shape(X)[0]
        X=tf.reshape(X, shape=(batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(X, perm=[0,2,1,3])
    
    def combine_heads(self, X):
        X=tf.transpose(X, perm=[0,2,1,3])
        batch_size=tf.shape(X)[0]
        return tf.reshape(X, shape=(batch_size, -1, self.num_heads, self.d_model))
    
    