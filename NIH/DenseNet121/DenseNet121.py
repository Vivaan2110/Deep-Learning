import tensorflow as tf 
import keras 
from preprocessing_denseNet import train_ds, test_ds, valid_ds, TRAIN_SIZE, TEST_SIZE, VALID_SIZE, weighted_bce

base_model=keras.applications.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3))

base_model.trainable=False # Uses the models pretrained weights

inputs=keras.layers.Input(shape=(224,224,3))

x=base_model(inputs, training=False)

x=keras.layers.GlobalAveragePooling2D()(x)
x=keras.layers.BatchNormalization()(x)

x=keras.layers.Dense(512, activation='elu')(x)
x = keras.layers.Dropout(0.3)(x)

outputs = keras.layers.Dense(14, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4), 
    loss=weighted_bce, 
    metrics=[keras.metrics.AUC(num_labels=14, name="auc", multi_label=True),
            keras.metrics.BinaryCrossentropy(name="binary_accuracy")])

earlyStop_cb=keras.callbacks.EarlyStopping(
    monitor="val_auc", 
    patience=3, 
    verbose=True, 
    restore_best_weights=True,
    mode="max")

history=model.fit(
    train_ds, 
    validation_data=valid_ds, 
    epochs=10, 
    steps_per_epoch=TRAIN_SIZE//32, 
    validation_steps=VALID_SIZE//32, 
    callbacks=[earlyStop_cb])

model.save("DenseNet121_Adam1e-4.keras")