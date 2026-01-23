This project implements a multi-label deep learning model for detecting thoracic diseases from chest X-ray images using the NIH Chest X-ray dataset.

It predicts the probability of 14 different pathologies from a single X-ray.

The system is built using TensorFlow and leverages transfer learning with DenseNet121. It also handles severe class imbalance using a custom weighted loss function.

Dataset:
NIH Chest X-ray Dataset
112,000 grayscale images
14 disease classes
Resized to 128×128 for custom CNN training and 224×224 for DenseNet121 inference

Model:
DenseNet121 backbone
Global Average Pooling
Fully connected head

Techniques Used:
tf.data pipelines
Disk caching
Early stopping
Single-image inference

Results:
Validation AUC = 0.81

Limitations:
Dataset contains severe class imbalances, which can affect rare disease performance
Trained on weakly labeled data
Performance depends on image quality

NOT intended for medical or clinical use.
