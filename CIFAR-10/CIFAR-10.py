import keras
import tensorflow as tf
from time import strftime
from pathlib import Path
from os import mkdir

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

model=keras.Sequential([
    keras.layers.Input(shape=(32,32,3),name='Input'),
    
    keras.layers.Conv2D(filters=32, kernel_initializer=he_init, kernel_size=(3,3),padding='same', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation='elu'),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.Conv2D(filters=32, kernel_initializer=he_init, kernel_size=(3,3),padding='same', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation='elu'),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    keras.layers.Conv2D(filters=64, kernel_initializer=he_init, kernel_size=(3,3),padding='same', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation='elu'),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.Conv2D(filters=64, kernel_initializer=he_init, kernel_size=(3,3),padding='same', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation='elu'),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=elu_act, kernel_initializer=he_init, kernel_regularizer=l2_reg),
    keras.layers.Dense(64,activation=elu_act,kernel_initializer=he_init, kernel_regularizer=l2_reg),
    keras.layers.Dense(10,activation='softmax')
    
])

SGD_optimizer=keras.optimizers.SGD(learning_rate=0.001, nesterov=True, momentum=0.9)

Adam_optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=Adam_optimizer,loss=loss,metrics=['accuracy'])

tensorboard_cb=keras.callbacks.TensorBoard(run_dir)

earlyStop_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3,verbose=1)

lr_cb=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=2,verbose=1, min_lr=1e-6)

history=model.fit(
    X_train,y_train,
    epochs=15,
    verbose=1,
    validation_data=(X_valid,y_valid),
    batch_size=32,
    callbacks=[tensorboard_cb,earlyStop_cb]
)

model.save("Saved Models/CNN_3x3_2ConvBlocks_ELU_Adam_BN_DO02.keras")