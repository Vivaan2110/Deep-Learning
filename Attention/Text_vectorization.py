import keras 
import tensorflow as tf 

d_model=256
seq_len=128

vocab=open('/Users/Vivaan/Documents/VS Code/Deep Learning/Attention/Text datasets/Shakespeare.txt').read()

vectorizer=keras.layers.TextVectorization(
    max_tokens=100,
    standardize=None,
    output_mode='int',
    split='character'
)

vectorizer.adapt([vocab])

tokens=vectorizer([vocab])[0]

dataset=tf.data.Dataset.from_tensor_slices(tokens)

dataset=dataset.batch(seq_len+1, drop_remainder=True)

def split(seq):
    return seq[:-1], seq[1:]

dataset=(
    dataset
    .shuffle(5000)
    .map(split)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

vocab_size=len(vectorizer.get_vocabulary())

inputs=keras.layers.Input(shape=(seq_len,))

embedding=keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
)(inputs)

PE=keras.layers.Embedding(
    input_dim=seq_len,
    output_dim=d_model
)

positions=tf.range(start=0, limit=seq_len, delta=1)

pos_encoding=PE(positions)

X=embedding+pos_encoding

for _ in range(6):
    MHA=keras.layers.MultiHeadAttention(
        num_heads=16,
        key_dim=16
    )(X,X,X,use_causal_mask=True)
    
    MHA=keras.layers.Dropout(0.125)(MHA)

    X=keras.layers.LayerNormalization()(X+MHA)
    
    feed_forw=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='gelu'),
        keras.layers.Dense(d_model)
    ])(X)
    
    feed_forw=keras.layers.Dropout(0.125)(feed_forw)
    
    X=keras.layers.LayerNormalization()(X+feed_forw)

logits=keras.layers.Dense(vocab_size)(X)

model=keras.Model(inputs=inputs,outputs=logits)

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)

adam=keras.optimizers.Adam(learning_rate=3e-4)

model.compile(
    optimizer=adam,
    loss=loss
)

model.fit(dataset, 
          epochs=40
)