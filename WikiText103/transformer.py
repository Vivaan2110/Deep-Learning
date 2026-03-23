import keras 
import tensorflow as tf 
from preprocessing import dataset, vocab_size, d_model, seq_len
from positional_encoding import PositionalEncoding
import json

tf.random.set_seed(0)

input=keras.layers.Input(shape=(seq_len, ))

emb=keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
)(input)

X=PositionalEncoding(
    d_model=d_model,
    seq_len=seq_len
)(emb)

def encoder_block(X):
    MHA=keras.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32
    )(X,X,X,use_causal_mask=True)
    
    X=keras.layers.LayerNormalization()(X+MHA)
    
    feed_forward=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='gelu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(d_model),
        keras.layers.Dropout(0.1)
    ])(X)
    
    X=keras.layers.LayerNormalization()(X+feed_forward)
    
    return X

N=8
for _ in range(N):
    X=encoder_block(X)

logits=keras.layers.Dense(vocab_size)(X)

model=keras.Model(inputs=input, outputs=logits)

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)

nadam=keras.optimizers.Nadam(learning_rate=1e-4, clipnorm=True)

model.compile(optimizer=nadam, loss=loss)

early_stop_cb=keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

lr_cb=keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    patience=2,
    factor=0.5
)

history=model.fit(
    dataset,
    epochs=30,
    callbacks=[early_stop_cb, lr_cb],
    steps_per_epoch=2000
)

model.save("wikitext103_dmodel256_seqlen64_numheads8_numlayers8_lrCB.keras")

with open('history.json','w') as f:
    json.dump(history.history, f)