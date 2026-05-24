import keras 
import numpy as np 
import tensorflow as tf 

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

def tversky_metric(true, pred):
    epsilon=1e-6
    true=tf.cast(true, tf.int32)
    pred=tf.cast(pred, tf.float32)
    
    pred=tf.argmax(pred, axis=-1)
    pred=tf.cast(pred, tf.int32)
    
    true=tf.one_hot(true, depth=4) # Metric required them to both be one hot encoded
    pred=tf.one_hot(pred, depth=4)
    
    true=tf.cast(true, tf.float32)
    pred=tf.cast(pred, tf.float32)
    
    axes=(0,1)
    
    TP=tf.reduce_sum(true*pred, axes)
    
    FP=tf.reduce_sum(pred*(1-true), axes)
    
    FN=tf.reduce_sum(true*(1-pred), axes)
    
    alpha=tf.constant(0.5, tf.float32)
    beta=tf.constant(0.5, tf.float32)
    
    numerator=TP+epsilon
    denominator=TP+alpha*FP+beta*FN+epsilon
    
    TM=numerator/denominator
    
    return tf.reduce_mean(TM)

print(tversky_metric(true, pred))