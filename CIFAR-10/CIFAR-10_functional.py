import keras
import tensorflow as tf
from time import strftime
from pathlib import Path
from os import mkdir

def lr_schedule(epoch, lr):
    base_lr=0.01
    if epoch<4:
        return base_lr
    elif epoch<8:
        return base_lr*0.5
    elif epoch<=10:
        return base_lr*0.025

def get_run_log_dir(root_dir='my_logs'):
    root_path=Path(root_dir)
    root_path.mkdir(exist_ok=True,parents=True)
    return Path(root_dir)/strftime("run_%Y_%m_%d_%H_%M_%S")

run_dir=get_run_log_dir()

cifar=keras.datasets.cifar10.load_data()

(X_train,y_train),(X_test,y_test)=cifar

X_train,X_test=X_train/255.0,X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)

elu_act=keras.activations.elu

he_init=keras.initializers.HeNormal()

l2_reg=keras.regularizers.l2()

data_aug=keras.Sequential([
    keras.layers.RandomFlip(mode='horizontal'),
    keras.layers.RandomTranslation(0.1,0.1),
])

inputs=keras.layers.Input(shape=(32,32,3),name='input')

x=data_aug(inputs)

x=keras.layers.Conv2D(32,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(32,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
block_1_output=keras.layers.MaxPool2D(pool_size=(2,2))(x)

x=keras.layers.Conv2D(64,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(block_1_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(64,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
skip_1=keras.layers.Conv2D(64,(1,1),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(block_1_output)
x=keras.layers.add([x,skip_1])
x=keras.layers.MaxPool2D(2,2)(x)
block_2_output=x

x=keras.layers.Conv2D(128,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(block_2_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(128,(3,3),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation=elu_act)(x)
x=keras.layers.Dropout(0.2)(x)
skip_2=keras.layers.Conv2D(128,(1,1),padding='same',kernel_regularizer=l2_reg, kernel_initializer=he_init)(block_2_output)
x=keras.layers.add([x,skip_2])
x=keras.layers.MaxPool2D(2,2)(x)
block_3_output=x

flatten=keras.layers.GlobalMaxPooling2D()(block_3_output)
x=keras.layers.Dense(128, activation=elu_act, kernel_initializer=he_init, kernel_regularizer=l2_reg)(flatten)
outputs=keras.layers.Dense(10, activation='softmax')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

SGD_optimizer=keras.optimizers.SGD(learning_rate=0.01, nesterov=True, momentum=0.9)

Adam_optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=SGD_optimizer,loss=loss,metrics=['accuracy'])

tensorboard_cb=keras.callbacks.TensorBoard(run_dir)

earlyStop_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=7,verbose=1, restore_best_weights=True)

lrPlateau_cb=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=2,verbose=1, min_lr=1e-5)

LrScedule_cb=keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

history=model.fit(
    X_train,y_train,
    epochs=40,
    verbose=1,
    validation_data=(X_valid,y_valid),
    batch_size=32,
    callbacks=[tensorboard_cb ,lrPlateau_cb, earlyStop_cb]
)

model.save("Saved Models/Residual_CNN_3x3_3ConvBlocks_ELU_SGD_BN_DO02_LearningRatePlateau_FlipSwitch.keras")