import tensorflow as tf 
import keras 
from transformer_class import TransformerEncoderStack, PatchEmbedding, PositionEmbedding

(X_train,y_train),(X_test,y_test)=keras.datasets.cifar10.load_data()

X_train,X_test=X_train/255.0, X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)

patch_size=4

d_model=64

num_heads=8

num_patches=(32//patch_size)**2

inputs=keras.layers.Input(shape=(32,32,3))

x=PatchEmbedding(patch_size, d_model)(inputs)
x=PositionEmbedding(num_patches, d_model)(x)

x=TransformerEncoderStack(
    num_heads=num_heads,
    num_layers=6,
    d_model=d_model,
    d_ff=4*d_model
)(x)

x=keras.layers.LayerNormalization()(x)
x=keras.layers.GlobalAveragePooling1D()(x)
x=keras.layers.Dense(128, activation='elu', kernel_initializer=keras.initializers.HeNormal(), kernel_regularizer=keras.regularizers.l2)(x)
outputs=keras.layers.Dense(10, activation='softmax')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    metrics=['accuracy'],
    loss=keras.losses.SparseCategoricalCrossentropy()
)

earlyStop_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=True, patience=3, restore_best_weights=True)

history=model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_valid, y_valid),
    verbose=True,
    callbacks=[earlyStop_cb]
)