import tensorflow as tf 
import keras 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 

root_dir='/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/data'

df=pd.read_csv('/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/data/Data_Entry_2017.csv')

all_labels=sorted(
    l for l in set("|".join(df["Finding Labels"]).split("|"))
    if l !="No Finding") # Finding all the labels 


label_to_index={l:i for i,l in enumerate(all_labels)} # Assigns index to the labels

# Creating a multi-hot encoding for each photo
def encode_labels(label_str):
    vec=[0]*len(all_labels)
    if label_str=="No Finding":
        return vec
    for l in label_str.split("|"):
        vec[label_to_index[l]]=1
    return vec

filename_to_path={}

for folder in os.listdir(root_dir):
    if folder.startswith("images_"):
        img_dir=os.path.join(root_dir,folder, "images")
        for fname in os.listdir(img_dir):
            filename_to_path[fname]=os.path.join(img_dir, fname)

labels=[] # Path to all teh label tensors
paths=[] # Path to all the image tensors

for _, row in df.iterrows():
    fname=row["Image Index"]
    if fname not in filename_to_path:
        continue
    
    paths.append(filename_to_path[fname])
    
    labels.append(encode_labels(row["Finding Labels"]))

paths=np.array(paths)
labels = np.array(labels, dtype=np.float32)

def load_image(path, label):
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img, channels=1)    
    img=tf.image.resize(img, (224,224))
    img=tf.image.grayscale_to_rgb(img) # DenseNet121 expects 3 channels
    img=tf.cast(img, tf.float32) / 255.0
    return img, label

def has_finding(image, label): # Returns True if the label has a finding
    return tf.reduce_sum(label)>0

full_dataset=tf.data.Dataset.from_tensor_slices((paths,labels))

full_dataset=(
    full_dataset
    .shuffle(len(paths),seed=0)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE) # TF decides how many CPU cores to use
)


TOTAL=len(paths)
TRAIN_SIZE=int(TOTAL*0.7)
TEST_SIZE=int(TOTAL*0.15)
VALID_SIZE=TOTAL-TEST_SIZE-TRAIN_SIZE

train_ds=(
    full_dataset
    .take(TRAIN_SIZE)
    .batch(32)
    .cache('/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/cache/train_cache')
    .repeat()
    .prefetch(tf.data.AUTOTUNE) # Loads the next batch while the current batch is still training
)

valid_ds=(
    full_dataset
    .skip(TRAIN_SIZE)
    .take(VALID_SIZE)
    .batch(32)
    .cache('/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/cache/valid_cache')
    .prefetch(tf.data.AUTOTUNE) 
)

test_ds=(
    full_dataset
    .skip(TRAIN_SIZE+VALID_SIZE)
    .take(TEST_SIZE)
    .batch(32)
    .cache('/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/cache/test_cache')
    .prefetch(tf.data.AUTOTUNE) 
)

pos_freq=labels.mean(axis=0) # Fraction of positives per class
neg_freq=1.0-pos_freq # Fraction of negatives per class

pos_weights=neg_freq/(pos_freq+1e-6)
pos_weights=pos_weights.astype(np.float32)

pos_weights_tf=tf.constant(pos_weights)

def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce=tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)
    weights=y_true*pos_weights_tf+(1.0-y_true)
    return tf.reduce_mean(bce*weights)