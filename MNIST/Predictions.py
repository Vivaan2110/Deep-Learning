import keras

digits=keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=digits

X_train,X_test=X_train/255.0,X_test/255.0

X_pred=X_test[:1000]

model=keras.models.load_model('/Users/Vivaan/Documents/VS Code/Deep Learning/MNIST/Saved models/Adam3e-4_l2reg1e-4_elu_heNormal.keras')

y_prob=model.predict(X_pred)

y_pred=y_prob.argmax(axis=1)

acc_list=(y_pred==y_test[:1000])

print(acc_list)