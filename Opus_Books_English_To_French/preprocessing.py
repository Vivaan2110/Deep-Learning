from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, Lowercase
import numpy as np
import tensorflow as tf

ds = load_dataset("opus_books", "en-fr")

seq_len=64
d_model=256

'''

def iterator():
    for row in ds['train']['translation']:
        yield row['en']
        yield row['fr']
        

tokenizer=Tokenizer(BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer=Whitespace() # This is a pre tokenizer wich removes whitespaces
tokenizer.normalizer=Sequence([Lowercase()]) # Normalises all the text to lowercase

trainer=BpeTrainer(
    vocab_size=12000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"] # BOS means beginning of sequence, EOS means End of sequence
)

tokenizer.train_from_iterator(iterator(), trainer=trainer)
tokenizer.save('tokenizer.json')

def format(row):
    en=row['en']
    fr=row['fr']
    return f"[BOS] English: {en} [EOS] French: {fr} [EOS]"

all_ids=[]

for row in ds['train']['translation']:
    text=format(row)
    ids=tokenizer.encode(text).ids
    all_ids.extend(ids)

np.save("IDS/all_ids.npy", np.array(all_ids, dtype=np.int32))

'''

all_ids=np.load("IDS/all_ids.npy")

tokenizer=Tokenizer.from_file('Tokenizer/tokenizer.json')

vocab_size=tokenizer.get_vocab_size()

def encode_pair(row):
    en: list=tokenizer.encode(row['en']).ids
    fr: list=tokenizer.encode(row['fr']).ids
    
    return en, fr

BOS=tokenizer.token_to_id("[BOS]")
EOS=tokenizer.token_to_id("[EOS]")

def preprocess(en_ids, fr_ids):
    enc=en_ids[:seq_len] # This is what the encoder will recieve
    
    dec_in=[BOS]+fr_ids[:seq_len-1] # This is what the decoder will recieve
    dec_out=fr_ids[:seq_len-1] + [EOS] # This is what the model will predict
    
    return enc, dec_in, dec_out

# Gives a fixed length to all the sequences
def pad(seq):
    return seq + [0]*(seq_len - len(seq))

enc_list = []
dec_in_list = []
dec_out_list = []

for row in ds['train']['translation']:
    en_ids, fr_ids = encode_pair(row)
    enc, dec_in, dec_out = preprocess(en_ids, fr_ids)

    enc_list.append(pad(enc))
    dec_in_list.append(pad(dec_in))
    dec_out_list.append(pad(dec_out))

enc_arr = tf.constant(enc_list, dtype=tf.int32)
dec_in_arr = tf.constant(dec_in_list, dtype=tf.int32)
dec_out_arr = tf.constant(dec_out_list, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices(
    ((enc_arr, dec_in_arr), dec_out_arr)
)

dataset = dataset.shuffle(10000).batch(8).prefetch(tf.data.AUTOTUNE)