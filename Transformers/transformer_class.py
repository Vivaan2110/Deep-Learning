import tensorflow as tf 
import keras 

# A transformer is 
# MultiHeadAttention -> Add and Norm -> Feed Forward Network -> Add and Norm -> Output

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.mha=keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        
        self.ffn=keras.Sequential([
            keras.layers.Dense(d_ff, activation='relu'),
            keras.layers.Dense(d_model)
        ])
        
        self.norm1=keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2=keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.drop1=keras.layers.Dropout(dropout)
        self.drop2=keras.layers.Dropout(dropout)
    
    def call(self, x, training=False):
        # Self-Attention
        
        attention_out=self.mha(x, x, x)
        attention_out=self.drop1(attention_out, training=training)
        x=self.norm1(x+attention_out) # Residual connection
        
        # Feed Forward Network
        feed_forward=self.ffn(x)
        feed_forward=self.drop2(feed_forward, training=False)
        x=self.norm2(x+feed_forward)
        
        return x

class TransformerEncoderStack(keras.layers.Layer):
    def __init__(self, num_heads, d_model, d_ff, num_layers):
        super().__init__()
        self.layers=[
            TransformerEncoder(num_heads=num_heads, d_model=d_model, d_ff=d_ff)
            for _ in range(num_layers)
        ]
    
    def call(self, x, training=False):
        for layer in self.layers:
            x=layer(x, training=False)
        
        return x

# Convert the 2D images to 1D vectors for the transformer
class PatchEmbedding(keras.layers.Layer): 
    def __init(self, patch_size, d_model):
        super().__init__()
        self.patch_size=patch_size
        self.proj=keras.layers.Dense(d_model)
    
    def call(self, images):
        patches=tf.image.extract_patches(
            images=images, # Returns a 4D tensor
            sizes=[1,self.patch_size,self.patch_size,1], # Determines the size of each patch
            strides=[1,self.patch_size,self.patch_size,1], # How far the window moves after extracting a patch
            rates=[1,1,1,1],
            padding="VALID" # Only extract patches that fit inside the image
        )
        
        batch_size=tf.shape(patches)[0] # Gets the dynamic batch_size
        patches=tf.reshape(patches,[batch_size-1,patches.shape[-1]])
        
        return self.proj(patches)