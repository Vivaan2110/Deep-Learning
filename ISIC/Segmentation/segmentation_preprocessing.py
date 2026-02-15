import tensorflow as tf 
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
    mask=tf.cast(mask>0, tf.float32)/255.0
    
    return mask

def image_preprocess(img_path, mask_path):
    tf.print("Reading:", img_path, mask_path)
    img=tf.io.read_file(img_path)
    img=tf.image.decode_jpeg(img, channels=3)
    img=tf.image.resize(img, (128,128))
    img=tf.cast(img, tf.float32)/255.0
    
    mask=mask_preprocess(mask_path)
    
    return img, mask

train_img, train_mask=get_image_mask_paths(TRAIN_PATH, MASK_PATH)
test_img, test_mask=get_image_mask_paths(TEST_PATH, MASK_PATH)
valid_img, valid_mask=get_image_mask_paths(VALID_PATH, MASK_PATH)

train_ds=tf.data.Dataset.from_tensor_slices((train_img, train_mask))
test_ds=tf.data.Dataset.from_tensor_slices((test_img, test_mask))
valid_ds=tf.data.Dataset.from_tensor_slices((valid_img, valid_mask))

train_ds=(
    train_ds
    .shuffle(2048,seed=0)
    .map(image_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache(CACHE_PATH+"/train")
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

for img, mask in train_ds.take(1):
    print(img.shape)
    print(mask.shape)