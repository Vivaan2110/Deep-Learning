import tensorflow as tf 
from PIL import Image 
import numpy as np
import os
import keras 

BASE_PATH='/Volumes/Extreme SSD/DRIVE retinal vessels/'
MASK_PATH=BASE_PATH+'training/1st_manual/'
IMG_PATH=BASE_PATH+'training/images/'

def get_img_mask_paths(img_path, mask_path):
    img_paths = []
    mask_paths = []

    for fname in sorted(os.listdir(img_path)):
        if fname.startswith("._") or not fname.endswith(".png"):
            continue

        img_full = os.path.join(img_path, fname)

        number = fname.split("_")[0]          # "21"
        mask_name = f"{number}_manual1.png"  # "21_manual1.png"
        mask_full = os.path.join(mask_path, mask_name)

        if not os.path.exists(mask_full):
            print("Missing mask:", mask_full)
            continue

        img_paths.append(img_full)
        mask_paths.append(mask_full)

    return img_paths, mask_paths

def mask_preprocess(path):
    mask=tf.io.read_file(path)
    mask=tf.image.decode_png(mask, channels=1)
    mask=tf.image.resize(mask, (288,288), 'nearest')
    
    mask=tf.cast(mask, tf.int32)
    
    mask=tf.cast((mask>0), tf.int32)
    
    return mask

def image_mask_preprocess(img_path, mask_path):
    img=tf.io.read_file(img_path)
    img=tf.image.decode_png(img)
    img=tf.image.resize(img, (288,288))
    
    img=tf.cast(img, tf.float32)/255.0
    
    mask=mask_preprocess(mask_path)
    
    return img, mask

img_paths, mask_paths=get_img_mask_paths(IMG_PATH, MASK_PATH)

ds=tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

ds=(
    ds
    .shuffle(buffer_size=len(img_paths))
)

TOTAL=len(img_paths)
TRAIN_SIZE=int(TOTAL*0.7)

train_ds=ds.take(TRAIN_SIZE)
valid_ds=ds.skip(TRAIN_SIZE)

def data_aug(img, mask):
    seed=tf.random.uniform([2], minval=0, maxval=1_000, dtype=tf.int32)
    
    seed1=seed+tf.constant([0,1])
    seed2=seed+tf.constant([0,2])
    seed3=seed+tf.constant([0,3])
    
    img=tf.image.stateless_random_flip_left_right(img, seed)
    mask=tf.image.stateless_random_flip_left_right(mask, seed)
    
    img=tf.image.stateless_random_flip_up_down(img, seed1)
    mask=tf.image.stateless_random_flip_up_down(mask, seed1)
    
    k = tf.random.stateless_uniform([], seed2, minval=0, maxval=4, dtype=tf.int32)
    
    img=tf.image.rot90(img, k)
    mask=tf.image.rot90(mask, k)
    
    return img, mask

train_ds=(
    train_ds
    .map(image_mask_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .map(data_aug, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(2)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .map(image_mask_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(2)
    .prefetch(tf.data.AUTOTUNE)
)

def dice_loss(y_true, y_pred):
    epsilon=1e-6
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_mean(y_true*y_pred, axis=axes)
    union=tf.reduce_mean(y_true, axes)+tf.reduce_mean(y_pred, axes)
    dice=2*intersection/(union+epsilon)
    return 1-tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    return 0.5*keras.losses.binary_crossentropy(y_true, y_pred)+2.0*dice_loss(y_true, y_pred)

def dice_metric(y_true, y_pred):
    epsilon=1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred>0.5, tf.float32)
    
    axes=(1,2,3) 
    
    intersection=tf.reduce_sum(y_true*y_pred,axis=axes)
    union=tf.reduce_sum(y_true,axis=axes)+tf.reduce_sum(y_pred,axis=axes)
    return tf.reduce_mean((2*intersection+epsilon)/(union+epsilon))