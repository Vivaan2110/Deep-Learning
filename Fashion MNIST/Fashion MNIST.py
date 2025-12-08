import tensorflow as tf 
import numpy as np
import keras

df=keras.datasets.fashion_mnist.load_data() # Load the dataset

(X_train,y_train),(X_test,y_test)=df

X_valid,y_valid=X_train[-5000:],y_train[-5000:]

X_train=X_train/255.0
X_test=X_test/255.0
X_valid=X_valid/255.0

class_names = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat", 
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]



# Building the model

#
# tf.random.set_seed(0) # have to set a random seed like in scikit learn to get reproducable results

# A sequential model is where
model=keras.Sequential(
    [
        # Flatten layer convert the input image into a 1D array, if it recieves [32,28,28] it converts it to [28,28]
        keras.layers.Flatten(input_shape=[28,28],name='flatten'), # First layer is a flatten layer which takes an input of (28x28)
        keras.layers.Dense(300,activation='relu',name='dense1'), # Number of neuron in the layer, its activation function is RELU and each layer should have a unique name
        keras.layers.Dense(100,activation='relu',name='dense2'),
        keras.layers.Dense(10,activation='softmax',name='output')
    ]
)

#print(model.summary()) # Prints a table showing the layer(type), output shape and parameters


'''
# Compiling the model

optimizer=keras.optimizers.SGD(learning_rate=0.1)  # The optimizer is called stochastic gradient descent
                                                    # The learning rate allows the model to update it parameters at every step
                                                    # A low learning rate doesnt let the model change the parameters fast to learn enough
                                                    # A high learning rate causes underfitting

loss=keras.losses.SparseCategoricalCrossentropy() # The loss function used is this as the output classes have sparse labels and are exclusive
                                                    # They have classes from 1-10


model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])

model.save("Saved models/sgd_learning_rate_0.1.keras") # After the model has been trained it can be saved



# Fitting the model

history = model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test))  # Epochs are the number of times the model sees the training data
                                                                                # The model updates weights after each epoch
                                                                                # The updation in weight directly depends on the learning rate
'''

# Load the saved model
model=keras.models.load_model("Saved models/sgd_learning_rate_0.1.keras")

X_new=X_test[10:15]

y_prob=model.predict(X_new)
y_pred=y_prob.argmax(axis=1)
print(y_pred)
print(y_test[10:15])

