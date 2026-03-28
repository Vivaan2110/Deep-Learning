import keras
import tensorflow as tf 
import numpy as np
from tokenizers import Tokenizer

from positional_encoding import PositionalEncoding

MODEL_PATH = 'Saved Models/OpusBooks_dmodel256_seqlen64_numheads16_numlayers4_lrCB.keras'
TOKENIZER_PATH = "Tokenizer/tokenizer.json"

tokenizer=Tokenizer.from_file(TOKENIZER_PATH)
model=keras.models.load_model(MODEL_PATH,
                              custom_objects={"PostionalEncoding":PositionalEncoding},
                              compile=False)

SEQ_LEN = 64
TEMPERATURE = 0.6
TOP_K = 5
MAX_NEW_TOKENS = 60

BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")
PAD_ID = tokenizer.token_to_id("[PAD]")
UNK_ID = tokenizer.token_to_id("[UNK]")

def top_k_sample(logits: tf.Tensor, k: int) -> int:
    values, indices = tf.math.top_k(logits, k=k)
    values = values.numpy()
    indices = indices.numpy()

    probs = tf.nn.softmax(values).numpy()
    next_id = np.random.choice(indices, p=probs)
    return int(next_id)


def translate(english_text: str,max_new_tokens: int = MAX_NEW_TOKENS,temperature: float = TEMPERATURE,top_k: int = TOP_K,) -> str:
    # Encoder input
    enc_ids = tokenizer.encode(english_text.lower()).ids[:SEQ_LEN]
    enc_ids = enc_ids + [PAD_ID] * (SEQ_LEN - len(enc_ids))

    # Decoder starts with BOS
    dec_ids = [BOS_ID]

    for _ in range(max_new_tokens):
        dec_context = dec_ids[-SEQ_LEN:]
        dec_context = dec_context + [PAD_ID] * (SEQ_LEN - len(dec_context))

        enc_tensor = tf.constant([enc_ids], dtype=tf.int32)
        dec_tensor = tf.constant([dec_context], dtype=tf.int32)

        logits = model([enc_tensor, dec_tensor])[:, len(dec_ids) - 1, :]
        logits = logits[0] / temperature

        # Ban PAD and UNK from generation
        logits = tf.tensor_scatter_nd_update(
            logits,
            indices=[[PAD_ID], [UNK_ID]],
            updates=[-1e9, -1e9]
        )

        next_id = top_k_sample(logits, k=top_k)

        if next_id == EOS_ID:
            break

        dec_ids.append(next_id)

    # Remove BOS before decoding
    french_ids = [tok for tok in dec_ids[1:] if tok not in (PAD_ID, EOS_ID)]
    return tokenizer.decode(french_ids)


if __name__ == "__main__":
    while True:
        english = input("English: ").strip()

        if english.lower() in {"quit", "exit"}:
            break

        french = translate(english)
        print("French:", french)
        print()