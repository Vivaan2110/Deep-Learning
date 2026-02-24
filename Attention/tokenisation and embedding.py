import keras
import tensorflow as tf

vocab={}

def tokenise(text: str):
    tokens=text.lower().split()
    ids=[]
    for t in tokens:
        if t not in vocab:
            vocab[t]=len(vocab) # Assigns a number based on how many numbers are in vocab
        ids.append(vocab[t])
    return ids

token_ids=tokenise("The man saw a woman a king and a queen")
token_ids=tf.convert_to_tensor(token_ids)

embedding=keras.layers.Embedding(
    input_dim=len(vocab), 
    output_dim=128
)

vectors=embedding(token_ids)