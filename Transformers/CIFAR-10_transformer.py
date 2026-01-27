import tensorflow as tf 
import keras 
from transformer_class import TransformerEncoderStack

(X_train,y_train),(X_test,y_test)=keras.datasets.cifar10.load_data()

X_train,X_test=X_train/255.0, X_test/255.0

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

X_train,y_train=X_train[:-5000],y_train[:-5000]

tf.random.set_seed(0)
print(X_train.shape)

#inputs=keras.layers.Input(shape=())

