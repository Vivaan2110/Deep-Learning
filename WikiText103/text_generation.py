import tensorflow as tf 
import sentencepiece as spm 
import numpy as np 
from preprocessing import seq_len
import keras
from positional_encoding import PositionalEncoding

sp=spm.SentencePieceProcessor()
sp.load('SPE Tokenizer/wikitext_tokenizer.model')

model=keras.models.load_model(
    '/Users/Vivaan/Documents/VS Code/Deep Learning/WikiText103/Saved Models/wikitext103_dmodel256_seqlen64_numheads8_numlayers8_lrCB.keras',
    custom_objects={"PositionalEncoding": PositionalEncoding},
    compile=False)


def generate_text(model, sp, prompt, seq_len, max_tokens=50, temperature=1.0, k=5): # Temperature contols randomness
    ids=sp.Encode(prompt, out_type=int) # Encode the prompt 
    
    for _ in range(max_tokens):
        context = ids[-seq_len:]

        if len(context) < seq_len:
            context = [0] * (seq_len - len(context)) + context
            
        x=tf.constant([context],dtype=tf.int32) # Convert the last 'seq_len' tokens into tensor
        
        logits=model(x)[:,-1,:] # Get the last token
        logits=logits/temperature
        
        unk_id=sp.unk_id()
        
        # Unknown id has a near 0 probablity
        logits=tf.tensor_scatter_nd_update(
            logits,
            indices=[[0,unk_id]],
            updates=[-1e-9]
        )
        
        # Adds a repetition penalty
        for token in set(ids):
            logits[0,token]/=1.2
        
        probs=tf.nn.softmax(logits)
        
        values, indices=tf.math.top_k(probs, k) # Only picks the top k elemets and returns its value and its position
        
        values=values[0].numpy()
        indices=indices[0].numpy()
        
        next_id=np.random.choice(indices, p=values/np.sum(values)) # Caclulates the probablity and outputs the best probablity
        ids.append(int(next_id))
        
    return sp.Decode(ids)

text=generate_text(model=model, sp=sp, seq_len=seq_len, prompt="The king said", k=20, temperature=0.7)

print(text)