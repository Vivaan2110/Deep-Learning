from segmentation_preprocessing import test_ds, combined_loss, dice_metric
import keras
import matplotlib.pyplot as plt
import PIL

model=keras.models.load_model(
    'Saved Models/Seg_ELU_L2_Adam_LrPlateau_CombinedLoss_DiceMetric.keras',
    custom_objects={
        "combined_loss":combined_loss,
        "dice_metric":dice_metric
    })

for img, mask in test_ds.take(20):
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