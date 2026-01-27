import tensorflow as tf 
import keras 
from DenseNet121.preprocessing_denseNet import test_ds, weighted_bce, all_labels

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

path='/Users/Vivaan/Downloads/d527ff6fc1482161c9225345c4ab42_big_gallery.jpg'

img=preprocess_img(path)

pred=model.predict(img)
probs=pred[0]

# Sorts in order and only prints top 5
pairs=sorted(zip(all_labels, probs), key=lambda x:x[1], reverse=True)[:5]

print("Top 5 Findings:")

for label,p in pairs:
    print(f"{label}: {p:.3f}")