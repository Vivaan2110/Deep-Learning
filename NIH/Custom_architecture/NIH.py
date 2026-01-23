import keras
import tensorflow as tf
from preprocessing import train_ds, test_ds, valid_ds, TRAIN_SIZE, TEST_SIZE, VALID_SIZE, weighted_bce

tf.random.set_seed(0)

he_init=keras.initializers.HeNormal()

elu_act=keras.activations.elu

l2_reg=keras.regularizers.l2(0.001)

data_aug=keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
])

inputs=keras.layers.Input(shape=(128,128,1), name='Input')

x=data_aug(inputs)

x=keras.layers.Conv2D(32, kernel_size=(5,5), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(32, kernel_size=(3,3), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
block_1_output=keras.layers.MaxPool2D((2,2))(x)

x=keras.layers.Conv2D(64, kernel_size=(3,3), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(block_1_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(64, kernel_size=(3,3), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
skip_layer_1=keras.layers.Conv2D(64, kernel_size=(1,1), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(block_1_output)
x=keras.layers.add([x, skip_layer_1])
x=keras.layers.MaxPool2D((2,2))(x)
block_2_output=x

x=keras.layers.Conv2D(128, kernel_size=(3,3), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(block_2_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(128, kernel_size=(3,3), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
skip_layer_2=keras.layers.Conv2D(128, kernel_size=(1,1), kernel_initializer=he_init, padding='same', kernel_regularizer=l2_reg)(block_2_output)
x=keras.layers.add([x,skip_layer_2])
x=keras.layers.MaxPool2D((2,2))(x)
block_3_output=x

x=keras.layers.GlobalAveragePooling2D()(block_3_output)
x=keras.layers.Dense(128, activation=elu_act, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Dense(64, activation=elu_act, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
outputs=keras.layers.Dense(14, activation='sigmoid')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

Adam_optimizer=keras.optimizers.Adam(0.0001, clipnorm=1.0)

SGD_optimizer=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.95, nesterov=True)

loss=keras.losses.BinaryCrossentropy()

model.compile(optimizer=Adam_optimizer, loss=weighted_bce, metrics=[keras.metrics.AUC(multi_label=True,num_labels=14,name="auc")
                                                        ,keras.metrics.BinaryAccuracy(name="binary_accuracy")])

lr_plateau=keras.callbacks.ReduceLROnPlateau(monitor="val_auc",patience=2, mode="max", factor=0.5, min_lr=1e-5, verbose=True)

earlyStop_cb=keras.callbacks.EarlyStopping(monitor="val_auc", patience=4, restore_best_weights=True, mode="max", verbose=True)

history=model.fit(train_ds, validation_data=valid_ds, epochs=10, callbacks=[earlyStop_cb], steps_per_epoch=TRAIN_SIZE//32, validation_steps=VALID_SIZE//32)

model.save("CNN_128_3Block_GAP_BN_DO02_L2_SGD_LrPlateau.keras")