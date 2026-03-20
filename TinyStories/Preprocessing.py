from datasets import load_dataset
import sentencepiece as spe 
import tensorflow as tf 
import numpy as np

ds = load_dataset("roneneldan/TinyStories")

d_model: int=int(256)
seq_len: int=int(128)

NUM_TRAIN_ROWS=2119719
NUM_VALID_ROWS=21990
'''
# Gets the text for each row in ds['train']
def sentence_iterator():
    for row in ds['train']:
        yield row['text']

spe.SentencePieceTrainer.train(
    sentence_iterator=sentence_iterator(),
    model_prefix='tinystories_tokenizer',
    vocab_size=5000,
    model_type='bpe', # BPE stands for byte pair encoder which splits the sentence into pairs
    character_coverage=1.0,
    max_sentence_length=10000,
    input_sentence_size=NUM_TRAIN_ROWS,
    shuffle_input_sentence=True
)'''

# Used to load the model
sp=spe.SentencePieceProcessor()
sp.load("tinystories_tokenizer.model")

vocab_size=sp.GetPieceSize()

print(vocab_size)

all_ids=np.load('/Users/Vivaan/Documents/VS Code/Deep Learning/TinyStories/IDs/all_ids.npy')

'''
# Convert the vocab to integer tokens as a stream to feed into the embedding layer
for row in ds['train']:
    ids=sp.Encode(row['text'], out_type=int)
    all_ids.extend(ids)
 
np.save("all_ids.npy", np.array(all_ids, dtype=np.int32)) 
'''

# First train on only the first 10 million tokens as it is very expensive
tokens=tf.convert_to_tensor(all_ids[:5000000])

dataset=tf.data.Dataset.from_tensor_slices(tokens)
dataset = dataset.batch(seq_len + 1, drop_remainder=True)

def split(seq):
    
    x,y = seq[:-1], seq[1:]
    
    # Makes sure the shape is correct
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