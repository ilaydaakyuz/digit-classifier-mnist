import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# 1. Veriyi Yükle
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize ve şekil değiştir
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 3. Data Augmentation tanımı
datagen = ImageDataGenerator(
    rotation_range=10,      # Hafif döndürme
    zoom_range=0.1,         # Yakınlaştırma
    width_shift_range=0.1,  # Sağa-sola kaydırma
    height_shift_range=0.1, # Yukarı-aşağı kaydırma
)

# 4. Model Mimarisi (Dropout ile)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),  # 🔒 Aşırı öğrenmeyi engeller
    tf.keras.layers.Dense(10, activation="softmax")
])

# 5. Modeli Derle
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 6. Eğitim
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=15
)

# 7. Modeli Kaydet
model.save("model.keras")

# 8. Eğitim sonrası örnek görüntüler (isteğe bağlı)
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(10,10,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")
plt.suptitle("Eğitimde kullanılan ilk 10 örnek")
plt.show()
