import keras 
import tensorflow as tf 
from os import mkdir
from time import strftime
from pathlib import Path

def get_run_logdir(root_logdir="my_logs"):
    root_path=Path(root_logdir)
    root_path.mkdir(parents=True,exist_ok=True)
    return Path(root_logdir)/strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir=get_run_logdir()

mnist=keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=mnist

X_train,X_test=X_train/255.0,X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)

elu_act=keras.activations.elu

he_init=keras.initializers.HeNormal()

model=keras.Sequential([
    keras.layers.Input(shape=(28,28,1),name='input'),
    
    # Filters is the number of features extracted
    # Increases as the layer gets deeper
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=elu_act,padding='same'),
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=elu_act,padding='same'),
    keras.layers.MaxPool2D(pool_size=(2,2)), # Pooling layers divides HxW by given value
    
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=elu_act,padding='same'),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=elu_act,padding='same'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    
    # All of the layers above are only for feature extraction
    # The next block will make decisions
    
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=elu_act),
    keras.layers.Dense(10,activation='softmax')
])

optimizer=keras.optimizers.Adam(learning_rate=3e-4,clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(  optimizer=optimizer,
                loss=loss,
                metrics=["accuracy"])

early_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,verbose=1,restore_best_weights=1)

tensorboard_cb=keras.callbacks.TensorBoard(run_logdir)

history=model.fit(
    X_train,y_train,
    epochs=16,
    verbose=1,
    validation_data=(X_valid,y_valid),
    batch_size=128,
    callbacks=[early_cb,tensorboard_cb]
)

model.save('Saved Models/Adam3e-4_elu.keras')