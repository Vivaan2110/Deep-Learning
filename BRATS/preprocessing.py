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

def convert_mask_to_labels(mask):
    c1 = mask[..., 0]
    c2 = mask[..., 1]
    c3 = mask[..., 2]

    label = tf.zeros_like(c1, dtype=tf.int32)

    label = tf.where(c1 > 0.5, 1, label)
    label = tf.where(c2 > 0.5, 2, label)
    label = tf.where(c3 > 0.5, 4, label)

    return label

def remap_labels(label):
    label = tf.where(label == 4, 3, label)
    return label

def load_h5(path):

    def _load(path_str):
        with h5py.File(path_str.decode(), 'r') as f:
            img = f['image'][:].astype(np.float32)
            mask = f['mask'][:].astype(np.float32)
        return img, mask

    img, mask = tf.numpy_function(
        _load,
        [path],
        [tf.float32, tf.float32]
    )

    img.set_shape([None, None, 4])
    mask.set_shape([None, None, 3])

    img = tf.image.resize(img, (128,128))
    mask = tf.image.resize(mask, (128,128), method='nearest')

    mean = tf.reduce_mean(img, axis=(0,1), keepdims=True)
    std = tf.math.reduce_std(img, axis=(0,1), keepdims=True)
    img = (img - mean) / (std + 1e-6)

    c1 = mask[..., 0]
    c2 = mask[..., 1]
    c3 = mask[..., 2]

    label = tf.zeros_like(c1, dtype=tf.int32)

    label = tf.where(c1 > 0.5, 1, label)
    label = tf.where(c2 > 0.5, 2, label)
    label = tf.where(c3 > 0.5, 4, label)

    # remap 4 -> 3
    label = tf.where(label == 4, 3, label)

    label = tf.expand_dims(label, axis=-1)

    return img, label

# Checks if the mask has a tumor or no
def has_tumor(img, mask):

    tumor_pixels = tf.reduce_sum(tf.cast(mask > 0, tf.int32))

    return tf.cond(

        tumor_pixels > 500,

        lambda: True,

        lambda: tf.random.uniform([]) < 0.1  # keep 10% empty

    )
    
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
    .repeat()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .map(load_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(has_tumor)
    .cache()
    .repeat()
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
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))

    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

    class_weights = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
    weighted_dice = dice * class_weights

    per_sample = tf.reduce_sum(weighted_dice, axis=-1) / tf.reduce_sum(class_weights)
    per_sample = tf.clip_by_value(per_sample, 0.0, 1.0)

    return 1.0 - tf.reduce_mean(per_sample)

def weighted_scce(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, tf.int32)

    ce = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )

    weights = tf.constant([0.001, 1.0, 3.0, 6.0])  # BG, 1,2,4
    pixel_weights = tf.gather(weights, y_true)

    return tf.reduce_mean(ce * tf.cast(pixel_weights, tf.float32))

def combined_loss(y_true, y_pred):
    return 0.5*weighted_scce(y_true, y_pred) + 1.5 * dice_loss(y_true, y_pred)

def dice_metric(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=4)
    
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true = y_true[..., 1:]

    y_pred = y_pred[..., 1:]

    axes = (1,2,3)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    return tf.reduce_mean((2 * intersection + 1e-6) / (union + 1e-6))


def dice_ET(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.cast(y_true == 3, tf.float32)
    y_pred = tf.cast(y_pred == 3, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true, axis=(1,2)) + tf.reduce_sum(y_pred, axis=(1,2))

    return tf.reduce_mean((2 * intersection + 1e-6) / (union + 1e-6))

def dice_TC(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.cast((y_true == 1) | (y_true == 3), tf.float32)
    y_pred = tf.cast((y_pred == 1) | (y_pred == 3), tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true, axis=(1,2)) + tf.reduce_sum(y_pred, axis=(1,2))

    return tf.reduce_mean((2 * intersection + 1e-6) / (union + 1e-6))

def dice_WT(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.cast(y_true > 0, tf.float32)
    y_pred = tf.cast(y_pred > 0, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true, axis=(1,2)) + tf.reduce_sum(y_pred, axis=(1,2))

    return tf.reduce_mean((2 * intersection + 1e-6) / (union + 1e-6))