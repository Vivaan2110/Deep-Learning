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

token_ids=tokenise("The cat ate a rat")
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

positions=tf.range(start=0, limit=token_ids.shape[0], delta=1) # Creates a tensor from 0 to seq_len-1 with step of 1

positions=tf.expand_dims(positions, axis=0)

X=vectors + PE(positions)

print(vectors.shape)
print(PE(positions).shape)