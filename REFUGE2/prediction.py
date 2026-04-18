import keras 
import tensorflow as tf 
from preprocessing import dice_metric, combined_loss, test_ds
import matplotlib.pyplot as plt

model=keras.models.load_model(
    "Saved Models/Seg_adamw_lr3e-4_combinedloss_weightedscce_dicemetric.keras",
    custom_objects={
        'combined_loss':combined_loss,
        'dice_metric':dice_metric
    }
)

for img, mask in test_ds.skip(10).take(1):
    break

pred=model.predict(img)

pred_mask = (pred[0] > 0.5).astype("float32")


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img[0,:,:,:3])  # show first channels if >3
plt.axis("off")

plt.show()

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(mask[0,:,:,0], cmap="gray")
plt.axis("off")

plt.show()

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()