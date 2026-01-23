import tensorflow as tf 
import keras 
from preprocessing_denseNet import test_ds, weighted_bce, all_labels

model=keras.models.load_model(
    '/Users/Vivaan/Documents/VS Code/Deep Learning/NIH/Saved Models/DenseNet121_Adam1e-4.keras', 
    custom_objects={"weighted_bce": weighted_bce}, 
    compile=False)

def preprocess_img(path):
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img, channels=1)
    img=tf.image.resize(img,(224,224))
    img=tf.image.grayscale_to_rgb(img)
    img=tf.cast(img, tf.float32)
    
    img=keras.applications.densenet.preprocess_input(img)
    img=tf.expand_dims(img, axis=0)
    return img

path='/Volumes/Extreme SSD/NIH dataset/kagglehub/datasets/nih-chest-xrays/data/images_002/images/00001373_018.png'

img=preprocess_img(path)

pred=model.predict(img)
probs=pred[0]

binary=(probs>=0.5).astype(int)

# Sorts in order adn only prints top 3
pairs=sorted(zip(all_labels, probs), key=lambda x:x[1], reverse=True)[:3]

for label,p in pairs:
    print(f"{label}: {p:.3f}")