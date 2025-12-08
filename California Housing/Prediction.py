import tensorflow as tf 
import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

housing=fetch_california_housing()

X=housing.data 
y=housing.target

X_full_train,X_test,y_full_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=5000, random_state=42)

X_to_pred=X_test[:1000]
model=keras.models.load_model("Saved Models/adam_learning_rate_0.00017.keras")

y_pred=model.predict(X_to_pred)

#print(y_test[:100])
#print(y_pred)

MSE=mean_squared_error(y_test[:1000],y_pred)

print(f"Mean squared error for each: {MSE:.2}")