import tensorflow as tf 
import keras 
import pandas as pd 
import numpy as np 
import os

BASE_PATH='/Volumes/Extreme SSD/ISIC-20/kagglehub/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset'

IMAGE_PATH=BASE_PATH+"/train"

df=pd.read_csv(BASE_PATH+'/ISIC_2020_Train_Metadata.csv')

label_dict=dict(zip(df["image_name"], df["target"]))

paths=[]
labels=[]

for name, label in label_dict:
    paths.append(os.path.join(IMAGE_PATH,name+".jpg"))
    labels.append(label)

paths=np.array(paths)
labels=np.array(labels)

def load_image(path, label):
    img=tf.io.read_file(path)
    img=tf.io.decode_jpeg(path)
    img=tf.image.resize(img, (128,128))
    img=tf.image.rgb_to_grayscale(img)
    img=tf.cast(img, tf.float32)/255.0
    
    return img, label

dataset=tf.data.Dataset.from_tensor_slices((paths,labels))

dataset=(
    dataset
    .shuffle(len(paths), seed=0)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

TOTAL=len(paths)
train_size=int(TOTAL*0.8)

train_ds=dataset.take(train_size)
valid_ds=dataset.skip(train_size)