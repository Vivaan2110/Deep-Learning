import numpy as np 
import keras 
import tensorflow as tf

vocab={}

seq_len: int = 100 # Number of input words the model can process
d_model:int = 128 # Number of dimensions of each processed word

def tokenise(text: str):
    tokens=text.lower().split()
    ids=[]
    
    for t in tokens:
        if t not in vocab:
            vocab[t]=len(vocab)
        ids.append(vocab[t])
    
    return ids 

token_ids=tokenise("The cat ate a rat and a bat")
token_ids=tf.constant([token_ids]) # Adds a batch dimension making it (batch, vocab_size, d_model)

vocab_size: int = len(vocab) # Total number of words known

embedding=keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
)

vectors=embedding(token_ids)

PE=keras.layers.Embedding(
    input_dim=seq_len,
    output_dim=d_model
)

positions=tf.range(start=0, limit=tf.shape(token_ids[0]), delta=1) # Creates a tensor from 0 to (input words)-1 with step of 1

positions=tf.expand_dims(positions, axis=0)

X=vectors + PE(positions)

# Create a function so it can be stacked multiple times
def encoder_block(X):

    attention=keras.layers.MultiHeadAttention( # num_heads*key_dims has to be d_model
        num_heads=8,
        key_dim=16
    )(X,X,X, use_causal_mask=True)
    
    X=keras.layers.LayerNormalization()(X+attention)

    feed_forward=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='relu'),
        keras.layers.Dense(d_model)
    ]
    )(X)

    X=keras.layers.LayerNormalization()(X+feed_forward)
    
    return X

N=12
# Stack the encoder block 12 times
for i in range(N):
    X=encoder_block(X)

logits=keras.layers.Dense(vocab_size)(X)
probs=tf.nn.softmax(logits)

print(logits.shape)
print(probs.shape)