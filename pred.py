from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


image_path = "pixil-frame-0.png"  

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

original_img = cv2.resize(img.copy(), (280, 280))  

img_resized = cv2.resize(img, (28, 28))
img_processed = 255 - img_resized
img_processed = img_processed / 255.0
img_processed = img_processed.reshape(1, 28, 28, 1)

model = tf.keras.models.load_model("model.keras")
predictions = model.predict(img_processed)
predicted_label = np.argmax(predictions)


plt.figure(figsize=(4, 4))
plt.imshow(original_img, cmap="gray")
plt.title(f"Tahmin: {predicted_label}", fontsize=16)
plt.axis("off")
plt.show()
