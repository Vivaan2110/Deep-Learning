import tensorflow as tf 
import keras
import numpy as np 

df=keras.datasets.fashion_mnist.load_data()

(X_train,y_train),(X_test,y_test)=df

model=keras.models.load_model("Saved models/sgd_learning_rate_0.1.keras")

X_new=X_test[0:10]
y_prob=model.predict(X_new)
y_pred=y_prob.argmax(axis=1)
print(y_pred)
print(y_test[0:10])