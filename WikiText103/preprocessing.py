from datasets import load_dataset
import sentencepiece as spe
import numpy as np
import tensorflow as tf 

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

NUM_ROWS=1801350

d_model=256
seq_len=64

'''
def sentence_iterator():
    for row in ds['train']:
        yield row['text']
        


spe.SentencePieceTrainer.train(
    model_type='bpe',
    sentence_iterator=sentence_iterator(),
    model_prefix='wikitext_tokenizer',
    vocab_size=4500,
    character_coverage=0.98,
    input_sentence_size=NUM_ROWS,
    shuffle_input_sentence=True,
    max_sentence_length=10000
)
'''

sp=spe.SentencePieceProcessor()
sp.Load("SPE Tokenizer/wikitext_tokenizer.model")

vocab_size=sp.GetPieceSize()

all_ids=np.load("IDS/all_ids.npy")

'''

for row in ds['train']:
    ids=sp.Encode(row['text'],out_type=int)
    all_ids.extend(ids)
        
np.save('all_ids.npy', np.array(all_ids, dtype=np.int32))
'''

tokens=tf.convert_to_tensor(all_ids)
dataset=tf.data.Dataset.from_tensor_slices(tokens)
dataset=dataset.batch(seq_len+1, drop_remainder=True)

def split(seq):
    x=seq[:-1]
    y=seq[1:]
    
    x=tf.ensure_shape(x, [seq_len])
    y=tf.ensure_shape(y, [seq_len])
    
    return x,y

dataset=(
    dataset
    .shuffle(5000)
    .map(split, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)