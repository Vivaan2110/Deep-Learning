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

def focal_tversky_loss(true, pred):
    epsilon=1e-6
    
    true=tf.cast(true, tf.int32)
    true=tf.one_hot(true, depth=4)
    
    true=tf.cast(true, tf.float32)
    pred=tf.cast(pred, tf.float32)
    
    axes=(0,1,2)
    
    TP=tf.reduce_sum(true*pred, axes)
    
    FP=tf.reduce_sum(pred*(1-true), axes)
    
    FN=tf.reduce_sum(true*(1-pred), axes)
    
    alpha=tf.constant(0.7, tf.float32)
    beta=tf.constant(0.3, tf.float32)
    gamma=tf.constant(4/3, tf.float32) # Larger the value of gamma the more the tougher regions are prioritised
                                       # The most used value is 4/3, after 2 the gradients startbecoming very large
    
    numerator=TP+epsilon
    denominator=TP+alpha*FP+beta*FN+epsilon
    
    TL=tf.reduce_mean(numerator/denominator)
    
    focal_TL=tf.math.pow(1-TL, 1/gamma) # Computes (1-TL)^(1/gamma)
    
    return focal_TL

print(focal_tversky_loss(true, pred))