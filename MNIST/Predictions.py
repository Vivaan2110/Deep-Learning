import keras
import numpy as np 
from handwritten_image_processing import preprocessImage, crop_image, deskew, dilate_pil
from pdf2image import convert_from_path
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

digits=keras.datasets.mnist.load_data()

(X_train,y_train),(X_test,y_test)=digits

X_train,X_test=X_train/255.0,X_test/255.0

X_pred=X_test[:1000]

# Converts each pdf page into a list of PIL images
pages=convert_from_path("Digits/Test_6.pdf")
img=pages[0]
img_cropped=crop_image(img)
img_dil = dilate_pil(img_cropped, kernel_size=3, iterations=2)
x = preprocessImage(img_dil, do_thicken=False, contrast_factor=1.0)   # or your preprocessImage


model=keras.models.load_model('Saved models/Adam3e-4_l2reg1e-4_elu_heNormal.keras')

#y_prob=model.predict(X_pred)
#y_pred=y_prob.argmax(axis=1)

#acc_list=(y_test[:1000]!=y_pred)

#print(acc_list)

probs = model.predict(x, verbose=0)[0]
print("top3:", [(int(i), float(probs[i])) for i in probs.argsort()[-3:][::-1]])
print("pred:", int(probs.argmax()))

# ---------- Diagnostics ----------
def diag_and_show(x_tensor, label="input"):
    a = x_tensor.reshape(28,28)
    print(f"=== {label} ===")
    print("shape,dtype:", x_tensor.shape, x_tensor.dtype)
    print("min,max,mean:", float(x_tensor.min()), float(x_tensor.max()), float(x_tensor.mean()))
    plt.figure(figsize=(2,2))
    plt.imshow(a, cmap='gray', vmin=0, vmax=1)
    plt.title(label)
    plt.axis('off')
    plt.show()

# your current tensor
# # from your pipeline
diag_and_show(x, "model input (current)")

# also show crop size and raw cropped image before preprocess
try:
    print("Cropped size (w,h):", img_cropped.size)
    plt.figure(figsize=(3,3)); plt.imshow(img_cropped.convert("L"), cmap='gray'); plt.axis('off'); plt.title("raw cropped"); plt.show()
except Exception as e:
    print("Could not show cropped image:", e)

# ---------- Quick PIL thickening fix ----------
def pil_thicken_and_predict(pil_img, passes=3, contrast_factor=2.0):
    im = pil_img.convert("L")
    for _ in range(passes):
        im = im.filter(ImageFilter.MaxFilter(3))
    im = ImageOps.autocontrast(im)
    im = ImageEnhance.Contrast(im).enhance(contrast_factor)

    # Save the **28×28 image before prediction**
    img_thickened = im.resize((28,28), Image.Resampling.LANCZOS)

    arr = np.array(img_thickened).astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    x2 = arr.reshape(1,28,28)

    return img_thickened, x2, model.predict(x2, verbose=0)[0]

print("\n-- Trying PIL thickening (3 passes, contrast 2.0) --")
img_thickened, x2, p2 = pil_thicken_and_predict(img_cropped, passes=3, contrast_factor=2.0)

#img_thickened: PIL.Image (28x28) — the image you showed
arr = np.array(img_thickened).astype(np.float32) / 255.0   # 0..1
# Ensure MNIST polarity: digit bright (1), background dark (0)
if arr.mean() > 0.5:
    arr = 1.0 - arr

x = arr.reshape(1,28,28).astype(np.float32)

print("tensor shape,dtype:", x.shape, x.dtype)
print("min,max,mean:", float(x.min()), float(x.max()), float(x.mean()))

plt.figure(figsize=(2,2))
plt.imshow(x.reshape(28,28), cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.show()

probs = model.predict(x, verbose=0)[0]
top3 = probs.argsort()[-3:][::-1]
print("top3 (digit,prob):", [(int(i), float(probs[i])) for i in top3])
print("predicted:", int(top3[0]))