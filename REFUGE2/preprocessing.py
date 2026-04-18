import tensorflow as tf 
from PIL import Image
import os
import keras

BASE_PATH='/Volumes/Extreme SSD/refuge2/versions/1/REFUGE2'
TRAIN_PATH=BASE_PATH+'/train'
VALID_PATH=BASE_PATH+'/val'
TEST_PATH=BASE_PATH+'/test'
CACHE_PATH=BASE_PATH+"/cache"

def get_image_mask_paths(img_path, mask_path):
    all_img_paths=[]
    all_mask_paths=[]
    
    for fname in sorted(os.listdir(img_path)):
        if fname.startswith("._"):
            continue
        
        if not fname.lower().endswith((".jpg",".png")):
            continue
        
        all_img_paths.append(os.path.join(img_path, fname))
        
        single_mask_path=fname[:-4]
        single_mask_path=single_mask_path+'.png'
        
        all_mask_paths.append(os.path.join(mask_path, single_mask_path))
        
    return all_img_paths, all_mask_paths

def mask_preprocess(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (288,288), method='nearest')
    
    mask = tf.cast(mask, tf.int32)

    mask = tf.where(mask == 255, 2, mask)
    mask = tf.where(mask == 128, 1, mask)
    mask = tf.where(mask == 0, 0, mask)

    mask = tf.cast(mask, tf.int32)

    return mask

def image_mask_preprocess(path, mask_path):
    
    img=tf.io.read_file(path)
    img=tf.image.decode_jpeg(img, channels=3)
    img=tf.image.resize(img, (288,288))
    img=tf.cast(img, tf.float32)/255.0
    
    mask=mask_preprocess(mask_path)
    
    return img, mask

train_img,train_mask=get_image_mask_paths(TRAIN_PATH+'/images', TRAIN_PATH+'/mask')
val_img,val_mask=get_image_mask_paths(VALID_PATH+'/images', VALID_PATH+'/mask')
train_img, train_mask=get_image_mask_paths(TRAIN_PATH+'/images', TRAIN_PATH+'/mask')

train_ds=tf.data.Dataset.from_tensor_slices((train_img, train_mask))

valid_ds=tf.data.Dataset.from_tensor_slices((val_img, val_mask))

test_ds=tf.data.Dataset.from_tensor_slices((train_img, train_mask))

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
    .shuffle(1000)
    .map(image_mask_preprocess, tf.data.AUTOTUNE)
    .repeat()
    .map(data_aug, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds=(
    valid_ds
    .map(image_mask_preprocess, tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds=(
    test_ds
    .map(image_mask_preprocess, tf.data.AUTOTUNE)
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

def dice_loss(y_true, y_pred):
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(y_true, depth=3)

    y_pred = tf.nn.softmax(y_pred)

    axes = (1, 2)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice = (2 * intersection + 1e-6) / (union + 1e-6)

    dice = dice[:, 1:]  

    return 1 - tf.reduce_mean(dice)

def weighted_scce(y_true, y_pred):
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    y_true=tf.cast(y_true, tf.int32)
    
    ce=keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # [background, disc, cup]
    weights=tf.constant([0.2, 1.0, 2.0])
    weights=tf.gather(weights, y_true)
    
    weighted_ce=ce*tf.cast(weights, tf.float32)
    
    return tf.reduce_mean(weighted_ce)

def combined_loss(y_true, y_pred):
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)        
    y_true = tf.cast(y_true, tf.int32)            

    ce = weighted_scce(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return ce + 2.0*dice

def dice_metric(y_true, y_pred):
    epsilon = 1e-6

    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
    y_pred = tf.nn.softmax(y_pred)

    axes = (1, 2, 3)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    return tf.reduce_mean((2 * intersection + epsilon) / (union + epsilon))

STEPS_PER_EPOCH=len(train_img)//8