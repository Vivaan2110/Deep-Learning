import keras 
import tensorflow as tf 

(X_train, y_train),(X_test,y_test)=keras.datasets.cifar10.load_data()

X_train,X_test=X_train/255.0,X_test/255.0

X_valid, y_valid=X_train[-5000:], y_train[-5000:]

X_train,y_train=X_train[:-5000],y_train[:-5000]

X_train, y_train=tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)

train_ds=tf.data.Dataset.from_tensor_slices((X_train, y_train))

def data_aug(img, label):
    seed=tf.random.uniform([2], maxval=1000, dtype=tf.int32)
    
    seed1=seed+tf.constant([1,0])
    seed2=seed+tf.constant([2,0])
    seed3=seed+tf.constant([3,0])
    seed4=seed+tf.constant([4,0])

    img=tf.image.stateless_random_flip_left_right(img, seed)
    
    img=tf.image.stateless_random_brightness(img, max_delta=0.15, seed=seed2)
    
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    img=tf.image.stateless_random_contrast(img, seed=seed3, lower=0.9, upper=1.1)
    
    k=tf.random.stateless_uniform([], seed=seed4, minval=0, maxval=5, dtype=tf.int32)
    
    img=tf.image.rot90(img, k)
    
    return img, label

train_ds=(
    train_ds
    .shuffle(1000)
    .map(data_aug, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

patch_size=2
d_model=128
image_size=32
num_patches=(image_size//patch_size)**2

tf.random.set_seed(0)

inputs=keras.layers.Input(shape=(32,32,3))

x=keras.layers.Conv2D(filters=d_model, kernel_size=patch_size, strides=patch_size)(inputs)
x=keras.layers.Reshape(target_shape=(num_patches, d_model))(x)

positions=tf.range(start=0, limit=num_patches)
pos_emb=keras.layers.Embedding(num_patches, d_model)(positions)

x=x+pos_emb

def encoder_block(x):
    attn=keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=d_model//4,
        dropout=0.1
    )(x, x)
    
    x=keras.layers.LayerNormalization()(x+attn)
    
    ffn=keras.Sequential([
        keras.layers.Dense(4*d_model, activation='gelu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(d_model),
        keras.layers.Dropout(0.1)
    ])(x)
    
    x=keras.layers.LayerNormalization()(x+ffn)
    
    return x

for _ in range(4):
    x=encoder_block(x)

x=keras.layers.GlobalAveragePooling1D()(x)
x=keras.layers.Dense(256, activation='gelu')(x)
x=keras.layers.Dropout(0.2)(x)
outputs=keras.layers.Dense(10, activation='softmax')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

AdamW=keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-4,clipnorm=1.0)

SCE=keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=AdamW,
    loss=SCE,
    metrics=['accuracy']
)

earlyStop_CB=keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    verbose=1,
    restore_best_weights=True,
    patience=3
)

lrPlateau_CB=keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=2,
    monitor='val_accuracy',
    verbose=1
)

history=model.fit(
    train_ds,
    batch_size=32,
    epochs=40,
    validation_data=(X_valid, y_valid),
    callbacks=[earlyStop_CB, lrPlateau_CB],
    verbose=1
)

model.save('Saved Models/ViT_dmodel128_patchsize2_numheads4_dataaug_lr1e-4.keras')