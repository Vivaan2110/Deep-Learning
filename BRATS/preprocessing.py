import h5py
import tensorflow as tf 
import os 
import numpy as np 
import keras

BASE='/Volumes/Extreme SSD/BraTS-2020 Segmentation'

IMAGE_FILE=BASE+'/BraTS Training/content/data'

CACHE_PATH=BASE+"/cache"

def data_aug(img, mask):
    seed=tf.random.uniform([2], maxval=10000, dtype=tf.int32)
    
    seed1=seed+tf.constant([1,0])
    seed2=seed+tf.constant([2,0])
    seed3=seed+tf.constant([3,0])
    seed4=seed+tf.constant([4,0])
    seed5=seed+tf.constant([5,0])
    
    img=tf.image.stateless_random_flip_left_right(img, seed1)
    mask=tf.image.stateless_random_flip_left_right(mask, seed1)
    
    img=tf.image.stateless_random_flip_up_down(img, seed2)
    mask=tf.image.stateless_random_flip_up_down(mask, seed2)
    
    k=tf.random.stateless_uniform([], seed3, maxval=4, dtype=tf.int32)
    
    img=tf.image.rot90(img, k)
    mask=tf.image.rot90(mask, k)
    
    return img, mask

def get_file_path(image_file):
    file_path=[]
    
    for fname in sorted(os.listdir(image_file)):
        
        if fname.startswith('._'):
            continue
        
        if not fname.endswith('.h5'):
            continue
        
        file_path.append(os.path.join(image_file,fname))
        
    return file_path


def load_h5(path): # Main function to be mapped
    
    # Tensorflow cant execute python file I/O so we have to make a helper function
    def _load(path_str):
        with h5py.File(path_str.decode(), 'r') as f:
            img=f['image'][:].astype(np.float32)
            mask=f['mask'][:].astype(np.float32)
        
        return img, mask
    
    img, mask=tf.numpy_function( # This is the bridge which wraps the numpy function and uses it as a tf op
        _load, # The function to run
        [path], # Input tensor
        [tf.float32, tf.float32] # Expected outputs
    )
    
    img.set_shape([128,128,4])
    mask.set_shape([128,128,3])
    
    epsilon=1e-6
    
    mean=tf.reduce_mean(img, axis=(0,1), keepdims=True)
    std=tf.math.reduce_std(img, axis=(0,1), keepdims=True)
    
    img=(img-mean)/(std+epsilon)
    
    img = tf.image.resize(img, (128,128), method='bilinear')
    mask = tf.image.resize(mask, (128,128), method='nearest')
    
    return img, mask

# Checks if the mask has a tumor or no
def has_tumor(img, mask):
    return tf.reduce_sum(mask)>0

paths=get_file_path(IMAGE_FILE)

np.random.shuffle(paths)

n = len(paths)

train = paths[:int(0.7 * n)]
valid = paths[int(0.7 * n):int(0.85 * n)]
test  = paths[int(0.85 * n):]

TRAIN_SIZE=len(train)
TEST_SIZE=len(test)
VALID_SIZE=len(valid)

train_ds=tf.data.Dataset.from_tensor_slices(train)
test_ds=tf.data.Dataset.from_tensor_slices(test)
valid_ds=tf.data.Dataset.from_tensor_slices(valid)

train_ds=(
    train_ds
    .map(load_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(has_tumor)
    .shuffle(256, seed=0)
    .map(data_aug, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .map(load_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(has_tumor)
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds=(
    test_ds
    .map(load_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(has_tumor)
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

def dice_loss(y_true, y_pred):
    epsilon=1e-6
    y_true=tf.cast(y_true, dtype=tf.float32)
    y_pred=tf.cast(y_pred, dtype=tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true*y_pred,axis=axes)
    union=tf.reduce_sum(y_true,axis=axes)+tf.reduce_sum(y_pred,axis=axes)
    
    dice=(2*intersection+epsilon)/(union+epsilon)
    
    return 1-tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    y_true=tf.cast(y_true, dtype=tf.float32)
    y_pred=tf.cast(y_pred, dtype=tf.float32)
    categorical=keras.losses.categorical_crossentropy(y_true, y_pred)
    dice=dice_loss(y_true, y_pred)
    
    return categorical+dice

def dice_metric(y_true, y_pred):
    epsilon=1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3)

    axes = (1, 2)
    
    intersection=tf.reduce_sum(y_true*y_pred,axis=axes)
    union=tf.reduce_sum(y_true,axis=axes)+tf.reduce_sum(y_pred,axis=axes)
    return tf.reduce_mean((2*intersection+epsilon)/(union+epsilon))

for img, mask in train_ds.take(1):
    print(tf.reduce_max(mask), tf.reduce_min(mask))
    print(np.unique(np.sum(mask, axis=-1)))