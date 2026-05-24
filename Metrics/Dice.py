import tensorflow as tf 
import keras 
import numpy as np 

y_true = np.array([
    [0,0,0],
    [1,2,3],
    [1,2,0]])

y_pred = np.array([
    [[0.1, 0.55, 0.25, 0.10],
        [0.90, 0.05, 0.03, 0.02],
        [0.85, 0.10, 0.03, 0.02]],
    
    [[0.10, 0.80, 0.05, 0.05],
        [0.05, 0.10, 0.75, 0.10],
        [0.05, 0.05, 0.10, 0.80]],
    
    [[0.15, 0.70, 0.10, 0.05],
        [0.10, 0.20, 0.60, 0.10],
        [0.88, 0.05, 0.04, 0.03],]
])
print(y_true.shape)
arg=tf.argmax(tf.cast(y_pred, tf.float32), axis=-1)
print(tf.one_hot(arg, depth=4))

def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true=tf.cast(y_true, tf.int32)
    y_true=tf.one_hot(y_true, depth=4) # One hot required for y_true unless it is binary segmentation
    y_pred=tf.cast(y_pred, tf.float32)
    
    y_pred=tf.one_hot(tf.argmax(y_pred, axis=-1), depth=4) 
     
    axes=(0,1)
     
    union=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    
    intersection=tf.reduce_sum(y_pred*y_true, axes)
    
    return tf.reduce_mean(2*intersection/(union+smooth))

print(dice_metric(y_true, y_pred))
    