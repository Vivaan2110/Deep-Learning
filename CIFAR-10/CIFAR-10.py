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

relu_act=keras.activations.relu

he_init=keras.initializers.HeNormal()

model=keras.Sequential([
    keras.layers.Input(shape=(32,32,3),name='Input'),
    
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation=relu_act,kernel_initializer=he_init,padding='same'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=relu_act,kernel_initializer=he_init),
    keras.layers.Dense(10,activation='softmax')
])

optimizer=keras.optimizers.Adam(learning_rate=1e-4,clipnorm=1)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

tensorboard_cb=keras.callbacks.TensorBoard(run_dir)

earlyStop_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3,verbose=1)

history=model.fit(
    X_train,y_train,
    epochs=15,
    verbose=1,
    validation_data=(X_valid,y_valid),
    batch_size=32,
    callbacks=[tensorboard_cb,earlyStop_cb]
)

model.save("Saved Models/Adam1e-4_relu_heNormal.keras")