import tensorflow as tf 
import keras
from preprocessing import seq_len, d_model, dataset, vocab_size
from positional_encoding import PositionalEncoding
import json

tf.random.set_seed(0)

# ENCODER ARCHITECTURE

Enc_input=keras.layers.Input(shape=(None, ), dtype=tf.int32)

emb=keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
)(Enc_input)

pos_emb=PositionalEncoding(
    d_model,
    seq_len
)(emb)

def encoder_block(X):
    MHA=keras.layers.MultiHeadAttention(
        num_heads=16,
        key_dim=16
    )(X, X, X)

    X=keras.layers.LayerNormalization()(X+MHA)
    
    feed_forw=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='gelu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(d_model),
        keras.layers.Dropout(0.1)
    ])(X)
    
    X=keras.layers.LayerNormalization()(X+feed_forw)
    
    return X

X=pos_emb

N=4
for _ in range(N):
    X=encoder_block(X)    

Enc_out=X

Dec_input=keras.layers.Input(shape=(None,), dtype=tf.int32)

decoder_emb=keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
    )(Dec_input)

decoder_pos_emb=PositionalEncoding(
    d_model,
    seq_len)(decoder_emb)

def decoder_block(X, Enc_out):
    MHA_masked=keras.layers.MultiHeadAttention(
        num_heads=16,
        key_dim=16
    )(X,X,X, use_causal_mask=True)
    
    X=keras.layers.LayerNormalization()(X+MHA_masked)
    
    MHA=keras.layers.MultiHeadAttention(
        num_heads=16,
        key_dim=16
    )(X, Enc_out, Enc_out)
    
    X=keras.layers.LayerNormalization()(X+MHA)
    
    feed_forw=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='gelu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(d_model),
        keras.layers.Dropout(0.1)
    ])(X)
    
    X=keras.layers.LayerNormalization()(X+feed_forw)
    
    return X

X=decoder_pos_emb

for _ in range(N):
    X=decoder_block(X, Enc_out)

Dec_out=keras.layers.Dense(vocab_size)(X)

model=keras.Model(inputs=[Enc_input, Dec_input], outputs=Dec_out)

nadam=keras.optimizers.Nadam(learning_rate=1e-4, clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=nadam, loss=loss)

early_stop_cb=keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

lr_cb=keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    patience=2,
    factor=0.5
)

history=model.fit(
    dataset,
    epochs=40,
    callbacks=[early_stop_cb, lr_cb],
    steps_per_epoch=1500
)

model.save("Saved Models/OpusBooks_dmodel256_seqlen64_numheads16_numlayers4_lrCB.keras")

with open('Saved Models/history.json','w') as f:
    json.dump(history.history, f)