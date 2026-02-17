import tensorflow as tf 
import tensorflow_addons as tfa 
import keras 
import numpy as np 
import os

BASE='/Volumes/Extreme SSD/ISIC Segmentation/versions/1'

TEST_PATH=BASE+"/ISIC2018_Task1-2_Test_Input"

TRAIN_PATH=BASE+"/ISIC2018_Task1-2_Training_Input"

VALID_PATH=BASE+"/ISIC2018_Task1-2_Validation_Input"

MASK_PATH=BASE+"/ISIC2018_Task1_Training_GroundTruth"

CACHE_PATH=BASE+"/cache"

def get_image_mask_paths(img_path, mask_path):
    img_paths=[]
    mask_paths=[]
    
    for fname in sorted(os.listdir(img_path)):
        if fname.startswith("._"):
            continue
        
        if not fname.lower().endswith((".jpg",".png")):
            continue
        
        img_paths.append(os.path.join(img_path,fname))
        
        mask_name=fname[:-4]
        
        mask_name=mask_name+"_segmentation.png"
        
        mask_paths.append(os.path.join(mask_path, mask_name))
    
    return img_paths, mask_paths

def mask_preprocess(path):
    mask=tf.io.read_file(path)
    mask=tf.image.decode_png(mask, channels=1)
    mask=tf.image.resize(mask, (128,128), method='nearest')
    mask=tf.cast(mask>0, tf.float32)
    
    return mask

def image_preprocess(img_path, mask_path):
    img=tf.io.read_file(img_path)
    img=tf.image.decode_jpeg(img, channels=3)
    img=tf.image.resize(img, (128,128))
    img=tf.cast(img, tf.float32)/255.0
    
    mask=mask_preprocess(mask_path)
    
    return img, mask

img, mask=get_image_mask_paths(TRAIN_PATH, MASK_PATH)

ds=tf.data.Dataset.from_tensor_slices((img, mask))

ds=(
    ds
    .shuffle(buffer_size=len(img), seed=0)
    
)

TOTAL=len(img)
TRAIN_SIZE=int(TOTAL*0.7)
VALID_SIZE=int(TOTAL*0.15)
TEST_SIZE=TOTAL-(TRAIN_SIZE+VALID_SIZE)

train_ds = ds.take(TRAIN_SIZE)
rest = ds.skip(TRAIN_SIZE)

valid_ds = rest.take(VALID_SIZE)
test_ds = rest.skip(VALID_SIZE)

def data_aug(img, mask):
    seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32) # Outputs randon number, controls sync between mask and image
    
    # Create differnt seeds so its more random
    seed1=seed+tf.constant([1,0])
    seed2=seed+tf.constant([2,0])
    seed3=seed+tf.constant([3,0])
    seed4=seed+tf.constant([4,0])
    seed5=seed+tf.constant([5,0])
    seed6=seed+tf.constant([6,0])
    
    # Stateless used as randomness only depends on seed
    img=tf.image.stateless_random_flip_left_right(img, seed)
    mask=tf.image.stateless_random_flip_left_right(mask, seed)
    
    img=tf.image.stateless_random_flip_up_down(img, seed1)
    mask=tf.image.stateless_random_flip_up_down(mask, seed1)
    
    # Creates a stateless version for ones that dont have stateless functions
    k = tf.random.stateless_uniform([], seed2, minval=0, maxval=4, dtype=tf.int32)
    
    img=tf.image.rot90(img, k)
    mask=tf.image.rot90(mask, k)
    
    img=tf.image.stateless_random_brightness(img, max_delta=0.15, seed=seed3) # Brightness only added on image as it can corrupt the mask
    img = tf.clip_by_value(img, 0.0, 1.0) # Used as brightness can increase pixel value beyond 1 which is invalid
    
    img=tf.image.stateless_random_contrast(img, seed=seed4, lower=0.9, upper=1.1)
    
    angle=tf.random.stateless_uniform([],seed5, minval=-0.2,maxval=0.2) # Image can be rotated randomly by 0.2 radians
    
    img=tfa.image.rotate(img, angle, interpolation='bilinear')
    mask=tfa.image.rotate(mask, angle, interpolation='nearest')
    
    dx=tf.random.stateless_uniform([], seed6, minval=-7, maxval=7) # X and Y can change differently 
    dy=tf.random.stateless_uniform([], seed6+tf.constant([1,0]), minval=-7, maxval=7)
    
    translations=tf.cast([dx,dy], tf.float32) # Casting as translate expects float translations
    
    img=tfa.image.translate(img, translations, interpolation='bilinear')
    mask=tfa.image.translate(mask, translations, interpolation='nearest')
    
    return img, mask


train_ds=(
    train_ds
    .shuffle(2048,seed=0)
    .map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(CACHE_PATH+"/train")
    .map(data_aug, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds=(
    test_ds
    .map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(CACHE_PATH+"/test")
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(CACHE_PATH+"/valid")
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

def dice_loss(y_true, y_pred):
    epsilon=1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    axes=(1,2,3) # Need to sum over (H,W,C) and not (batch, H, W ,C)
    
    intersection=tf.reduce_sum(y_true*y_pred,axis=axes)
    union=tf.reduce_sum(y_true,axis=axes)+tf.reduce_sum(y_pred,axis=axes)
    dice=(2*intersection+epsilon)/(union+epsilon)
    
    return 1-tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)+dice_loss(y_true, y_pred)

def dice_metric(y_true, y_pred):
    epsilon=1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred>0.5, tf.float32)
    
    axes=(1,2,3) 
    
    intersection=tf.reduce_sum(y_true*y_pred,axis=axes)
    union=tf.reduce_sum(y_true,axis=axes)+tf.reduce_sum(y_pred,axis=axes)
    return tf.reduce_mean((2*intersection+epsilon)/(union+epsilon))