import tensorflow as tf 
import keras
from Preprocessing import train_ds, valid_ds, pos_weights, valid_size, train_size

he_init=keras.initializers.HeNormal()

elu=keras.activations.elu

l2_reg=keras.regularizers.l2(1e-4)

tf.random.set_seed(0)

data_aug=keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(factor=(0.1)),
    keras.layers.RandomTranslation(height_factor=0.05,width_factor=0.05),
    keras.layers.RandomZoom(0.2)
])

inputs=keras.layers.Input(shape=(128,128,3))

x=data_aug(inputs)

x=keras.layers.Conv2D(32, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(32, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(32, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
block_1_output=keras.layers.MaxPool2D((2,2))(x)

x=keras.layers.Conv2D(64, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_1_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(64, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(64, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
skip_layer_1=keras.layers.Conv2D(64, kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_1_output)
x=keras.layers.add([skip_layer_1,x])
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.MaxPool2D((2,2))(x)
block_2_output=x

x=keras.layers.Conv2D(128, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_2_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(128, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(128, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
skip_layer_2=keras.layers.Conv2D(128, kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_2_output)
x=keras.layers.add([skip_layer_2,x])
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.MaxPool2D((2,2))(x)
block_3_output=x

x=keras.layers.Conv2D(256, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_3_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(256, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
x=keras.layers.Conv2D(256, kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu)(x)
skip_layer_3=keras.layers.Conv2D(256, kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_init, kernel_regularizer=l2_reg)(block_3_output)
x=keras.layers.add([x,skip_layer_3])
x=keras.layers.MaxPool2D((4,4))(x)
x=keras.layers.Activation(activation=elu)(x)
block_4_output=x

x=keras.layers.GlobalAveragePooling2D()(block_4_output)
x=keras.layers.Dense(256, activation=elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Dropout(0.4)(x)
x=keras.layers.Dense(128, activation=elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Dropout(0.4)(x)
outputs=keras.layers.Dense(1, activation='sigmoid')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

adam=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

sgd=keras.optimizers.SGD(learning_rate=5e-4, momentum=0.95, nesterov=True)

loss=keras.losses.BinaryCrossentropy()

model.compile(
    optimizer=sgd,
    loss=loss,
    metrics=[keras.metrics.AUC(name='auc'),keras.metrics.Precision(thresholds=0.2), keras.metrics.Recall(thresholds=0.2)]
)

class_weight={
    0:1.0,
    1:float(pos_weights)
}

earlyStop_cb=keras.callbacks.EarlyStopping(monitor='val_auc',patience=5, restore_best_weights=True, verbose=1, mode='max')

lrPlateau=keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, verbose=1)

history=model.fit(
    train_ds,
    validation_data=valid_ds,
    batch_size=32,
    class_weight=class_weight,
    epochs=15,
    verbose=1,
    callbacks=[earlyStop_cb,lrPlateau]
)

model.save("Saved Models/4CNN_block_Res_SGD_M_N_Plateau.keras")