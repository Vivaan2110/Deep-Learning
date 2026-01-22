import keras 
import tensorflow as tf 

mnist=keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=mnist

X_train,X_test=X_train/255.0,X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

tf.random.set_seed(0)

print(X_train.shape)

l2_reg=keras.regularizers.l2(0.001)

inputs=keras.layers.Input(shape=(28,28,1), name='input')

x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=l2_reg)(inputs)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation='elu')(x)
x=keras.layers.Dropout(rate=0.2)(x)
x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation='elu')(x)
x=keras.layers.Dropout(rate=0.2)(x)
block_1_output=keras.layers.MaxPool2D(pool_size=(2,2))(x)

x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=l2_reg)(block_1_output)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation='elu')(x)
x=keras.layers.Dropout(rate=0.2)(x)
x=keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=l2_reg)(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation='elu')(x)
x=keras.layers.Dropout(rate=0.2)(x)
block_2_output=keras.layers.add([x,block_1_output])


flatten=keras.layers.Flatten()(block_2_output)
x=keras.layers.Dense(128, activation='elu', kernel_regularizer=l2_reg)(flatten)
x=keras.layers.Dense(64, activation='elu', kernel_regularizer=l2_reg)(x)
outputs=keras.layers.Dense(10, activation='softmax')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.Adam(learning_rate=3e-4,clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(  optimizer=optimizer,
                loss=loss,
                metrics=["accuracy"])

early_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,verbose=1,restore_best_weights=1)

history=model.fit(
    X_train,y_train,
    epochs=16,
    verbose=1,
    validation_data=(X_valid,y_valid),
    batch_size=32,
    callbacks=[early_cb]
)

model.save("Saved Models/ResCNN_2Block_BN_ELU.keras")