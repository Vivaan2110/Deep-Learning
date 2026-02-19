import h5py
import tensorflow as tf 
import os 
import numpy as np 

BASE='/Volumes/Extreme SSD/BraTS-2020 Segmentation'

IMAGE_FILE=BASE+'/BraTS Training/content/data'

CACHE_PATH=BASE+"/cache"

def data_aug(img, mask):
    seed=tf.random.uniform([], maxval=10000, dtype=tf.int32)
    
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
    
    img=tf.image.stateless_random_brightness(img, max_delta=0.15, seed=seed4)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    img=tf.image.stateless_random_contrast(img, seed=seed5, lower=0.9, upper=1.1)
    
    return img, mask

def get_file_path(image_file):
    file_path=[]
    
    for fname in sorted(os.listdir(image_file)):
        
        if not(fname.endswith('.h5')):
            continue
        
        file_path.append(os.path.join(image_file,fname))
        
    return file_path


def load_h5(path): # Main function to be mapped
    
    # Tensorflow cant execute python file I/O so we have to make a helper function
    def _load(path_str):
        with h5py.File(path_str.decode(), 'r') as f:
            img=f['image'][:]
            mask=f['mask'][:]
        
        return img, mask
    
    img, mask=tf.numpy_function( # This is the bridge which wraps the numpy function and uses it as a tf op
        _load, # The function to run
        [path], # Input tensor
        [tf.float32, tf.float32] # Expected outputs
    )
    
    img.set_shape([240,240,4])
    mask.set_shape([240,240,3])
    
    return img, mask

file_paths=get_file_path(IMAGE_FILE)

dataset=tf.data.Dataset.from_tensor_slices(file_paths)

TOTAL=len(file_paths)
TRAIN_SIZE=TOTAL*0.7
TEST_SIZE=TOTAL*0.15
VALID_SIZE=TOTAL-(TRAIN_SIZE+TEST_SIZE)

dataset=(
    dataset
    .map(load_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=TOTAL, seed=0)
)

train_ds=dataset.take(TRAIN_SIZE)
rest=dataset.skip(TRAIN_SIZE)

valid_ds=rest.take(VALID_SIZE)
test_ds=rest.skip(VALID_SIZE)

train_ds=(
    train_ds
    .cache(CACHE_PATH+"/train")
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .cache(CACHE_PATH+"/valid")
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds=(
    test_ds
    .cache(CACHE_PATH+"/test")
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)