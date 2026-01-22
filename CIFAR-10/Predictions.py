import keras

model=keras.models.load_model("Saved Models/CNN_3x3_2ConvBlocks_ELU_BN_DO02.keras")

cifar=keras.datasets.cifar10.load_data()

(X_train,y_train),(X_test,y_test)=cifar

X_train,X_test=X_train/255.0,X_test/255.0

y_prob=model.predict(X_test[0:20])
y_pred=y_prob.argmax(axis=1)

print(y_pred)
print(y_test[0:20])