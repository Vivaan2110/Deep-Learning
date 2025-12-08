import tensorflow as tf 
import keras
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from pathlib import Path
from time import strftime
from os import mkdir

def get_run_logdir(root_logdir='my_logs'):
    root_path=Path(root_logdir)
    root_path.mkdir(exist_ok=True,parents=True)
    return Path(root_logdir)/strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir=get_run_logdir()

tensorboard_callback=keras.callbacks.TensorBoard(run_logdir,profile_batch=(100,200))

housing=fetch_california_housing()

X=housing.data
y=housing.target

X_full_train,X_test,y_full_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=5000, random_state=42)

tf.random.set_seed(0)

# The normalization layer acts like a standard scaler and scales down the values 
# It does this by using the mean and the variance which it learns via the adaprt method 
norm_layer=keras.layers.Normalization(input_shape=(8,))
norm_layer.adapt(X_train)

model=keras.Sequential([
    norm_layer,                                 # This is the input layer which converts all inputs to 2 ranges
    keras.layers.Dense(50,activation="relu",kernel_regularizer=keras.regularizers.l2(0.0001)),  # An l2 regularizer adds penalties to larger weights 
                                                                                                # This stops the model from memorising data
                                                                                                # The 0.0001 controls how strong the penalty is
                                                                                                # A larger value forces the weights to be small
    keras.layers.Dense(50,activation="relu",kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(50,activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(1)                       # Output layer is only 1 neuron as only 1 output, house price is being calculated   
] 
)

optimizer=keras.optimizers.Adam(learning_rate=0.00017)   # Adam is an optimzed opimizer for regressions problems 
                                                        # It automatically adjusts the learning rate per parameter
                                                        # Regression often has noise which is handled much better by adam
                                                        # Requires little to no tuning



model.compile(
    loss="mse",
    optimizer=optimizer,
    metrics=['RootMeanSquaredError']
)


# Creates a callback function which stops the training when the monitored metric stops increasing
# Paience is the total number of epochs that that the model needs to have worse results
early=keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min'
)


history=model.fit(  X_train,y_train,
                    epochs=20,
                    validation_data=(X_valid,y_valid),
                    callbacks=[early,tensorboard_callback])

model.save("Saved Models/adam_learning_rate_0.00017.keras")