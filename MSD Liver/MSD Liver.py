from preprocessing import train_ds, val_ds, combined_loss, dice_metric, dice_liver, dice_bg, dice_tumour, STEPS_PER_EPOCH, VAL_STEPS
import tensorflow as tf 
import keras 
import numpy as np 

tf.random.set_seed(0)

he_init=keras.initializers.HeNormal()

l2_reg=keras.regularizers.l2(5e-4)

gelu_act=keras.activations.gelu

inputs=keras.layers.Input(shape=(64, 64, 16, 1))

x=keras.layers.Conv3D(32, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(inputs)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(32, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
block_1_output=x
x=keras.layers.MaxPool3D((2,2,1))(block_1_output)

skip_layer_1=keras.layers.Conv3D(64, (1,1,1), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Conv3D(64, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(64, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Add()([x, skip_layer_1])
block_2_output=x
x=keras.layers.MaxPool3D((2,2,1))(x)

skip_layer_2=keras.layers.Conv3D(128, (1,1,1), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Conv3D(128, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(128, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Add()([x, skip_layer_2])
block_3_output=x
x=keras.layers.MaxPool3D((2,2,1))(x)

x=keras.layers.Conv3D(256, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Dropout(0.15)(x)

x=keras.layers.Conv3D(256, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Dropout(0.15)(x)

bottleneck=x

x=keras.layers.Conv3DTranspose(128, kernel_size=(2,2,2), strides=(2,2,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(bottleneck)
x=keras.layers.Concatenate()([x, block_3_output])
x=keras.layers.Dropout(0.15)(x)

block_1_decoder_output=keras.layers.Conv3D(128, kernel_size=(1,1,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)

x=keras.layers.Conv3D(128, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(128, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Add()([x, block_1_decoder_output])

x=keras.layers.Conv3DTranspose(64, kernel_size=(2,2,2), strides=(2,2,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Concatenate()([x, block_2_output])
x=keras.layers.Dropout(0.15)(x)

block_2_decoder_output=keras.layers.Conv3D(64, kernel_size=(1,1,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)

x=keras.layers.Conv3D(64, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(64, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Add()([x, block_2_decoder_output])

x=keras.layers.Conv3DTranspose(32, kernel_size=(2,2,1), strides=(2,2,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Concatenate()([x, block_1_output])
x=keras.layers.Dropout(0.15)(x)

block_3_decoder_output=keras.layers.Conv3D(32, kernel_size=(1,1,1), padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)

x=keras.layers.Conv3D(32, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Conv3D(32, (3,3,3), (1,1,1), "same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(gelu_act)(x)
x=keras.layers.Add()([x, block_3_decoder_output])

outputs=keras.layers.Conv3D(3, kernel_size=(1,1,1), activation='softmax')(x)

model=keras.Model(inputs, outputs)

adamW=keras.optimizers.AdamW(1e-4, 1e-4, clipnorm=1.0)

sgd=keras.optimizers.SGD(1e-3,momentum=0.95, nesterov=True)

model.compile(
    optimizer=adamW,
    loss=combined_loss,
    metrics=[dice_metric, dice_bg, dice_liver, dice_tumour]
)

earlyStop_cb=keras.callbacks.EarlyStopping(
    monitor='val_dice_metric',
    patience=5,
    verbose=1,
    restore_best_weights=True,
    mode='max'
)

lrPlateau_cb=keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.3,
    verbose=1,
    min_lr=5e-7,
    mode='min'
)

history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=13,
    callbacks=[earlyStop_cb, lrPlateau_cb],
    steps_per_epoch = 1000,
    validation_steps = 100
)

model.save('Saved Models/3D_Conv_AdamW_lr1e-4_LrPlateau_CombinedLoss_DiceMetric_DicePerClass.keras')