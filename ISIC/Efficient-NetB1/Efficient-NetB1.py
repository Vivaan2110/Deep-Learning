import tensorflow as tf 
import keras 
import sys

sys.path.append("/Users/Vivaan/Documents/VS Code/Deep Learning/ISIC/Efficient-NetB1")

from Preprocessing_Eff import train_ds, valid_ds, pos_weights, valid_size, train_size

tf.random.set_seed(0)

model=keras.models.load_model('Saved Models/EfficientNetB1_Adam_Plateau_FineTuneSgdModel_1e-5_40UnfrozenLayers.keras', compile=False)

'''
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
'''

# Finds all the BN layers and disables them as they de-stabablise fine tuning
for layer in model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable=False

he_init=keras.initializers.HeNormal()

elu=keras.activations.elu

l2_reg=keras.regularizers.l2(1e-4)

'''
base_model=keras.applications.EfficientNetB1(
    weights="imagenet",
    input_shape=(240,240,3),
    include_top=False
)

base_model.trainable=False
'''

base_model=model.get_layer('efficientnetb1')

base_model.trainable=True

for layer in base_model.layers[:-80]:
    layer.trainable = False

'''
inputs=keras.layers.Input(shape=(240,240,3))

x=base_model(inputs, training=False)

x=keras.layers.GlobalAveragePooling2D()(x)
x=keras.layers.Dense(256, activation=elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Dropout(0.4)(x)
x=keras.layers.Dense(128, activation=elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)(x)
x=keras.layers.Dropout(0.4)(x)
outputs=keras.layers.Dense(1, activation='sigmoid')(x)

model=keras.Model(inputs=inputs, outputs=outputs)
'''

adam=keras.optimizers.Adam(learning_rate=5e-6, clipnorm=1.0)

sgd=keras.optimizers.SGD(learning_rate=5e-4, momentum=0.95, nesterov=True)

loss=keras.losses.BinaryCrossentropy()

model.compile(
    optimizer=adam,
    loss=loss,
    metrics=[keras.metrics.AUC(name='auc'),keras.metrics.Precision(thresholds=0.2), keras.metrics.Recall(thresholds=0.2)]
)

class_weight={
    0:1.0,
    1:float(pos_weights)
}

earlyStop_cb=keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5, 
    restore_best_weights=True, 
    verbose=1, 
    mode='max'
)

lrPlateau=keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", 
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint=keras.callbacks.ModelCheckpoint(
    filepath="Saved Models/EfficientNetB1_Adam_Plateau_2ndFineTune_5e-6_80UnfrozenLayers.keras",
    verbose=True, 
    monitor="val_auc",
    save_best_only=True,
    mode="max"
)

history=model.fit(
    train_ds,
    validation_data=valid_ds,
    batch_size=32,
    class_weight=class_weight,
    epochs=10,
    verbose=1,
    callbacks=[earlyStop_cb,lrPlateau, checkpoint]
)

model.save("Saved Models/EfficientNetB1_Adam_Plateau_2ndFineTune_5e-6_80UnfrozenLayers.keras")
