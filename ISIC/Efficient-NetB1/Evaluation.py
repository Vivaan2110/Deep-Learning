import tensorflow as tf 
import keras 
import sys

sys.path.append("/Users/Vivaan/Documents/VS Code/Deep Learning/ISIC/Efficient-NetB1")

from Preprocessing_Eff import train_ds, valid_ds, pos_weights, valid_size, train_size

model=keras.models.load_model('Saved Models/EfficientNetB1_Adam_Plateau_2ndFineTune_5e-6_80UnfrozenLayers.keras')

model.evaluate(valid_ds)