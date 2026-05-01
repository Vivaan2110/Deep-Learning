import keras
import tensorflow as tf
from preprocessing import train_ds, valid_ds, TRAIN_SIZE, VALID_SIZE, dice_metric, combined_loss, dice_ET, dice_TC, dice_WT

tf.random.set_seed(0)

he_init=keras.initializers.HeNormal()

elu_act=keras.activations.elu

l2_reg=keras.regularizers.l2(1e-4)

inputs=keras.layers.Input(shape=(128,128,4))

# Encoder which downscales the image
x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(inputs)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
block_1_output=x
x=keras.layers.MaxPool2D((2,2))(x)

skip_layer_1=keras.layers.Conv2D(filters=64, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x,skip_layer_1])
block_2_output=x
x=keras.layers.MaxPool2D((2,2))(x)

skip_layer_2=keras.layers.Conv2D(filters=128, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x,skip_layer_2])
block_3_output=x
x=keras.layers.MaxPool2D((2,2))(x)

skip_layer_3=keras.layers.Conv2D(filters=256, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x,skip_layer_3])
block_4_output=x
x=keras.layers.MaxPool2D((2,2))(x)

x=keras.layers.Conv2D(512, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Dropout(0.1)(x)

x=keras.layers.Conv2D(512, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Dropout(0.1)(x)

bottleneck=x

# Decoder which upscales the images

x=keras.layers.Conv2DTranspose(filters=256,kernel_size=(2,2),strides=2, padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(bottleneck)
x=keras.layers.Concatenate()([x,block_4_output])
x=keras.layers.Dropout(0.2)(x)

block_1_decoder_output=keras.layers.Conv2D(256, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)

x=keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x, block_1_decoder_output])

x=keras.layers.Conv2DTranspose(filters=128,kernel_size=(2,2),strides=2, padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Concatenate()([x,block_3_output])
x=keras.layers.Dropout(0.2)(x)

block_2_decoder_output=keras.layers.Conv2D(128, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)

x=keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x, block_2_decoder_output])

x=keras.layers.Conv2DTranspose(filters=64,kernel_size=(2,2),strides=2, padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Concatenate()([x,block_2_output])
x=keras.layers.Dropout(0.2)(x)

block_3_decoder_output=keras.layers.Conv2D(64, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)

x=keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x, block_3_decoder_output])

x=keras.layers.Conv2DTranspose(filters=32,kernel_size=(2,2),strides=2, padding="same", kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Concatenate()([x,block_1_output])
x=keras.layers.Dropout(0.2)(x)

block_4_decoder_output=keras.layers.Conv2D(32, kernel_size=(1,1), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)

x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer=he_init, kernel_regularizer=l2_reg, padding="same")(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(elu_act)(x)
x=keras.layers.add([x, block_4_decoder_output])

outputs=keras.layers.Conv2D(4, kernel_size=(1,1), activation='softmax')(x)

model=keras.Model(inputs, outputs)

adam=keras.optimizers.Adam(1e-4, clipnorm=1.0)

sgd=keras.optimizers.SGD(1e-3,momentum=0.95, nesterov=True)

model.compile(
    optimizer=adam,
    loss=combined_loss,
    metrics=[dice_metric, dice_ET, dice_TC, dice_WT]
)

earlyStop_cb=keras.callbacks.EarlyStopping(
    monitor='val_dice_metric',
    patience=10,
    verbose=1,
    restore_best_weights=True,
    mode='max'
)

lrPlateau_cb=keras.callbacks.ReduceLROnPlateau(
    monitor='val_dice_metric',
    patience=3,
    factor=0.5,
    verbose=1,
    min_lr=5e-7,
    mode='max'
)

history=model.fit(
    train_ds,
    batch_size=32,
    epochs=13,
    callbacks=[earlyStop_cb, lrPlateau_cb],
    validation_data=valid_ds,
    steps_per_epoch = 1500,
    validation_steps = 100
)

model.save('Saved Models/Seg_Adam_lr1e-4_LrPlateau_CombinedLoss_DiceMetric_DicePerClass.keras')