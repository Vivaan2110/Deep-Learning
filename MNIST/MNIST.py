import tensorflow as tf 
import keras 
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

# Removes the last 5000 from the training set so no leaks occur
X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)

l2_reg=keras.regularizers.l2(l2=1e-4)

elu_act=keras.activations.elu

# An initializer sets the weights before training
he_init=keras.initializers.HeNormal()

model=keras.Sequential([
    keras.layers.Flatten(input_shape=[28,28],name='Input'),
    keras.layers.Dense(128,activation=elu_act,name='Dense1',kernel_initializer=he_init),
    keras.layers.Dense(64,activation=elu_act,name='Dense2',kernel_initializer=he_init),
    keras.layers.Dense(10,activation='softmax',name='Output')
])

optimizer=keras.optimizers.Adam(learning_rate=3e-4,clipnorm=1.0)

loss=keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

early_cb=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,verbose=True,restore_best_weights=True)

tensorboard_cb=keras.callbacks.TensorBoard(run_logdir)

history=model.fit(X_train,y_train,epochs=20,verbose=True,validation_data=(X_valid,y_valid),batch_size=128,callbacks=[early_cb,tensorboard_cb])

print("Last 5 train acc:", history.history["accuracy"][-5:])
print("Last 5 val acc:", history.history["val_accuracy"][-5:])

val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
print("Eval on valid:", val_acc)

#model.save('Saved Models/Adam3e-4_l2reg1e-4_elu_heNormal.keras')