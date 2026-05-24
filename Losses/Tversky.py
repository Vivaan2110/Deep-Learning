import tensorflow as tf 
import keras 
import numpy as np

true = np.array([
    [0,0,0],
    [1,2,3],
    [1,2,0]])

pred = np.array([
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

def tversy_loss(true, pred):
    epslion=1e-6
    true=tf.cast(true, tf.int32)
    true=tf.one_hot(true, depth=4)
    
    true=tf.cast(true, tf.float32)
    pred=tf.cast(pred, tf.float32)
    
    axes=(0,1,2)
    
    TP=tf.reduce_sum(true*pred, axes) # Predicitng where lesion exists
    
    FP=tf.reduce_sum(pred*(1-true), axes) # Predicting where lesion does not exist
    
    FN=tf.reduce_sum((1-pred)*true,axes) # Not predicting where lesion exists
    
    alpha=tf.constant(0.9, tf.float32) # Weight for penalising FP
    
    beta=tf.constant(0.1, tf.float32) # Weight for penalising FN
    
    if tf.reduce_sum(alpha)+tf.reduce_sum(beta)!=tf.constant(1.0, tf.float32):
        raise ValueError("Sum of alhpa and beta should be 1")
    
    numerator=TP+epslion
    denominator=TP+alpha*FP+beta*FN+epslion
    
    TL=numerator/denominator
    
    return 1-tf.reduce_mean(TL)

print(tversy_loss(true, pred))