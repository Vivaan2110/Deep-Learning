import nibabel as nib 
import tensorflow as tf 
import os 
import keras 
import numpy as np
from tqdm import tqdm

BASE_PATH='/Users/Vivaan/.cache/Datasets/MSD Liver'
IMG_PATH=BASE_PATH+'/imagesTr'
MASK_PATH=BASE_PATH+'/labelsTr'
TFRECORD_PATH=os.path.join(BASE_PATH, "msd_liver_patches_0.25organ_0.75tumour.tfrecord")

def get_img_mask_path(img_path, mask_path):
    all_img_paths=[]
    all_mask_paths=[]
    
    for fname in sorted(os.listdir(IMG_PATH)):
        if fname.startswith('._'):
            continue
        
        if not(fname.endswith('.nii.gz')):
            continue
    
        all_img_paths.append(os.path.join(img_path, fname))
        all_mask_paths.append(os.path.join(mask_path, fname))
    
    return all_img_paths, all_mask_paths

# Is like normalising a normal image but in CT scans its in terms of Housenfielf Units(HU)
def normalise_ct(img, min_hu=-100, max_hu=400):
    img = np.clip(img, min_hu, max_hu)

    img = (img - min_hu) / (max_hu - min_hu)

    return img.astype(np.float32)

# Creates patches where the organ or tumour is present
def create_foreground_patches(img, mask, patch_size=(64, 64, 16)):
    ph, pw, pd=patch_size
    coords=np.argwhere(mask>0) # Pixels where mask has value 1 or 2
    
    if len(coords) == 0:

        return None, None
    
    random_index=tf.random.uniform([], 0, tf.shape(coords)[0], dtype=tf.int32) # Gives a random value from 0 to the total number of foreground voxel (3D equivalent of a pixel)
    center = coords[np.random.randint(len(coords))] # gets one random voxel from coords
    
    ch, cw, cd=center[0], center[1], center[2] # H, W, D for center
    
    H, W, D=img.shape # Shape of the image
    
    h=np.clip(ch-ph//2, 0, H-ph) # Start position, Minimum start index, Max start index
    w=np.clip(cw-pw//2, 0, W-pw)
    d=np.clip(cd-pd//2, 0, D-pd)
    
    img_patch=img[h:h+ph, w:w+pw, d:d+pd] # Starts from start postion and ends at start position+patch_size
    mask_patch = mask[h:h+ph, w:w+pw, d:d+pd] 
    
    img_patch = img_patch[..., np.newaxis]
    
    return img_patch.astype(np.float32), mask_patch.astype(np.int32)

def create_tumour_patches(img, mask, patch_size=(64, 64, 16)):
    ph, pw, pd=patch_size
    coords=np.argwhere(mask>1)
    
    if len(coords) == 0:
        return None, None
    
    center=coords[np.random.randint(len(coords))]
    
    ch, cw, cd=center[0], center[1], center[2]
    
    H, W, D=img.shape
    
    h=np.clip(ch-ph//2, 0, H-ph)
    w=np.clip(cw-pw//2, 0, W-pw)
    d=np.clip(cd-pd//2, 0, D-pd)
    
    img_patch=img[h:h+ph, w:w+pw, d:d+pd]
    mask_patch = mask[h:h+ph, w:w+pw, d:d+pd] 
    
    img_patch = img_patch[..., np.newaxis]
    
    return img_patch.astype(np.float32), mask_patch.astype(np.int32)
    
def serialize_patch(img_patch, mask_patch):
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_patch.tobytes()])
        ),
        "mask": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[mask_patch.tobytes()])
        ),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )

    return example.SerializeToString()

def create_tfrecord(img_paths, mask_paths, output_path, patches_per_volume=16):
    with tf.io.TFRecordWriter(output_path) as writer:
        count = 0

        for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
            img = nib.load(img_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.int32)

            img = normalise_ct(img)

            print(os.path.basename(img_path), np.unique(mask))

            for _ in range(4):
                img_patch, mask_patch = create_foreground_patches(img,mask,(64,64,16))

                if img_patch is None:
                    continue

                serialized = serialize_patch(img_patch, mask_patch)
                writer.write(serialized)

                count += 1
            
            for _ in range(12):
                img_patch, mask_patch = create_tumour_patches(img,mask,(64,64,16))

                if img_patch is None:
                    continue

                serialized = serialize_patch(img_patch, mask_patch)
                writer.write(serialized)

                count += 1

    print("Saved patches:", count)
    
    

img, mask=get_img_mask_path(IMG_PATH, MASK_PATH)

#create_tfrecord(img,mask,TFRECORD_PATH,patches_per_volume=16)

def parse_tfrecord(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_raw(example["image"], tf.float32)
    mask = tf.io.decode_raw(example["mask"], tf.int32)

    image = tf.reshape(image, (64, 64, 16, 1))
    mask = tf.reshape(mask, (64, 64, 16))

    return image, mask

TOT=len(img)

TRAIN_SIZE=int(TOT*0.85)
VALID_SIZE=int(TOT*(0.15))

STEPS_PER_EPOCH=TRAIN_SIZE//2
VAL_STEPS=VALID_SIZE//2

ds=tf.data.TFRecordDataset(TFRECORD_PATH)

train_ds=ds.take(TRAIN_SIZE)
val_ds=ds.skip(TRAIN_SIZE)

train_ds=(
    train_ds
    .shuffle(512)
    .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(2)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds=(
    val_ds
    .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(2)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

def dice_loss(y_true, y_pred):
    epsilon=1e-6
    
    y_true=tf.cast(y_true, tf.int32)
    y_true=tf.one_hot(y_true, depth=3)
    
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    union=tf.reduce_sum(y_true*y_pred, axes)
    
    dice=2*union/(intersection+epsilon)
    
    weights=tf.constant([0.5, 2.0, 10.0], dtype=tf.float32)
   
    weighted_dice=weights*dice 
    
    per_sample = tf.reduce_sum(weighted_dice, axis=-1) / tf.reduce_sum(weights)
    per_sample = tf.clip_by_value(per_sample, 0.0, 1.0)

    return 1.0 - tf.reduce_mean(per_sample)
def weighted_scce(y_true, y_pred):
    y_true=tf.cast(y_true, tf.int32)
    
    y_pred=tf.cast(y_pred, tf.float32)
    
    weights=tf.constant([0.5, 1.0, 20.0], dtype=tf.float32)
    
    scce=keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    voxel_weights=tf.gather(weights, y_true)
    
    return tf.reduce_mean(scce*tf.cast(voxel_weights, tf.float32))

def combined_loss(y_true, y_pred):
    return 0.3*weighted_scce(y_true, y_pred)+dice_loss(y_true, y_pred)

def dice_bg(y_true, y_pred):
    epsilon=1e-6
    
    y_true=tf.cast(y_true, tf.int32)
    y_pred=tf.argmax(y_pred, axis=-1)
    
    y_true=tf.cast(y_true==0, tf.float32)
    y_pred=tf.cast(y_pred==0, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    union=tf.reduce_sum(y_true*y_pred, axes)
    
    dice=2*union/(intersection+epsilon)
    
    return tf.reduce_mean(dice)

def dice_liver(y_true, y_pred):
    epsilon=1e-6
    
    y_true=tf.cast(y_true, tf.int32)
    y_pred=tf.argmax(y_pred, axis=-1)
    
    y_true=tf.cast(y_true==1, tf.float32)
    y_pred=tf.cast(y_pred==1, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    union=tf.reduce_sum(y_true*y_pred, axes)
    
    dice=2*union/(intersection+epsilon)
    
    return tf.reduce_mean(dice)

def dice_tumour(y_true, y_pred):
    epsilon=1e-6
    
    y_true=tf.cast(y_true, tf.int32)
    y_pred=tf.argmax(y_pred, axis=-1)
    
    y_true=tf.cast(y_true==2, tf.float32)
    y_pred=tf.cast(y_pred==2, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    union=tf.reduce_sum(y_true*y_pred, axes)
    
    dice=2*union/(intersection+epsilon)
    
    return tf.reduce_mean(dice)
    
    
def dice_metric(y_true, y_pred):
    epsilon=1e-6
    
    y_true=tf.cast(y_true, tf.int32)
    y_pred=tf.argmax(y_pred, axis=-1)
    
    y_true=tf.one_hot(y_true, depth=3)

    y_pred=tf.one_hot(y_pred, depth=3)
    
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred, tf.float32)
    
    axes=(1,2,3)
    
    intersection=tf.reduce_sum(y_true, axes)+tf.reduce_sum(y_pred, axes)
    union=tf.reduce_sum(y_true*y_pred, axes)
    
    dice=2*union/(intersection+epsilon)
    
    dice=dice[...,1:]
    
    return tf.reduce_mean(dice)
