import tensorflow as tf 
import keras 

x=tf.constant(((0, 1, 2),(3 ,4 ,5)), dtype=tf.float32, shape=(2,3))
y=tf.reduce_sum(x, axis=0)

a=tf.constant((1,2,3, 4), dtype=tf.float32)
b=tf.constant((4,5,6), dtype=tf.float32)

z=tf.split(a, axis=0, num_or_size_splits=2)
print(tf.nn.softmax(x))