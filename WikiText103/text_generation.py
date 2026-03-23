import tensorflow as tf 
import sentencepiece as spm 
import numpy as np 

sp=spm.SentencePieceProcessor()
sp.load('SPE Tokenizer/wikitext_tokenizer.model')

model=1

def generate_text(model, sp, prompt, seq_len, max_tokens=50, temperature=1.0, k=5): # Temperature contols randomness
    ids=sp.Encode(prompt, out_type=int) # Encode the prompt 
    
    for _ in range(max_tokens):
        x=tf.constant([ids[-seq_len:]],dtype=tf.int32) # Convert the last 'seq_len' tokens into tensor
        
        logits=model(x)[:,-1,:] # Get the last token
        logits=logits/temperature
        
        probs=tf.nn.softmax(logits)
        
        values, indices=tf.math.top_k(probs, k) # Only picks the top k elemets and returns its value and its position
        
        value=values[0].numpy()
        indices=indices[0].numpy()
        
        next_id=np.random.choice(indices, p=values/np.sum(values)) # Caclulates the probablity and outputs the best probablity
        ids.append(next_id)
        
    return sp.Decode(int(next_id))

text=generate_text(model=model, sp=sp, prompt="The king said", k=5)