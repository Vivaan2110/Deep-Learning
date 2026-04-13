import tensorflow as tf 
from tokenizers import Tokenizer
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, Lowercase
import numpy as np
import os

seq_len=64
d_model=128

TOKENIZER_DIR = "Tokenizer"
IDS_DIR = "IDS"

TRAIN_PATH = os.path.join(IDS_DIR, "train_pairs.npz")
VALID_PATH = os.path.join(IDS_DIR, "valid_pairs.npz")

TRAIN_SIZE = 50000
VALID_SIZE = 3000

ds=load_dataset("wmt14", "fr-en")

train_split = ds["train"].select(range(TRAIN_SIZE))
valid_split = ds["validation"].select(range(VALID_SIZE))

def iterator():
    for row in ds['train']['translation']:
        yield row['en']
        yield row['fr']

tokenizer=Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer=ByteLevel()
tokenizer.normalizer=Sequence([Lowercase()])

trainer=BpeTrainer(
    vocab_size=25000,
    min_frequency=3,
    special_tokens=["[EOS]","[BOS]","[PAD]","[UNK]"]
)

tokenizer.train_from_iterator(iterator=iterator(), trainer=trainer)

tokenizer.save("Tokenizer/tokenizer.json")

vocab_size=tokenizer.get_vocab_size()

BOS=tokenizer.token_to_id("[BOS]")
EOS=tokenizer.token_to_id("[EOS]")
PAD = tokenizer.token_to_id("[PAD]")

def pad(seq):
    return seq + [PAD]*(seq_len - len(seq))

def build_arrays(split):
    enc_list = []
    dec_in_list = []
    dec_out_list = []

    for row in split["translation"]:
        en_ids = tokenizer.encode(row["en"]).ids
        fr_ids = tokenizer.encode(row["fr"]).ids

        enc = en_ids
        dec_in = [BOS] + fr_ids[:seq_len - 1]
        dec_out = fr_ids[:seq_len - 1] + [EOS]

        enc_list.append(pad(enc))
        dec_in_list.append(pad(dec_in))
        dec_out_list.append(pad(dec_out))

    return (
        np.array(enc_list, dtype=np.int32),
        np.array(dec_in_list, dtype=np.int32),
        np.array(dec_out_list, dtype=np.int32),
    )

train_enc, train_dec_in, train_dec_out = build_arrays(train_split)
valid_enc, valid_dec_in, valid_dec_out = build_arrays(valid_split)

np.savez_compressed(
    TRAIN_PATH,
    enc=train_enc,
    dec_in=train_dec_in,
    dec_out=train_dec_out
)

np.savez_compressed(
    VALID_PATH,
    enc=valid_enc,
    dec_in=valid_dec_in,
    dec_out=valid_dec_out
)