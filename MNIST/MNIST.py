import tensorflow as tf 
import keras 

mnist=keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=mnist

X_train,X_test=X_train/255.0,X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

# Removes the last 5000 from the training set so no leaks occur
X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)

l2_reg=keras.regularizers.l2(l2=1e-4)

model=keras.Sequential([
    keras.layers.Flatten(input_shape=[28,28],name='Input'),
    keras.layers.Dense(300,activation='relu',kernel_regularizer=l2_reg,name='Dense1'),
    keras.layers.Dense(100,activation='relu',kernel_regularizer=l2_reg,name='Dense2'),
    keras.layers.Dense(10,activation='softmax',name='Output')
])

optimizer=keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

early_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,verbose=True,restore_best_weights=True)

history=model.fit(X_train,y_train,epochs=20,verbose=True,validation_data=(X_valid,y_valid),batch_size=128,callbacks=[early_cb])