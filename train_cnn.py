import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# =========================
# LOAD DATASET MNIST
# =========================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =========================
# NORMALIZE DATA
# =========================

x_train = x_train / 255.0
x_test = x_test / 255.0

# =========================
# RESHAPE FOR CNN
# (60000,28,28,1)
# =========================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# =========================
# BUILD CNN MODEL
# =========================

model = Sequential([

    # CONVOLUTION LAYER 1
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    MaxPooling2D((2,2)),


    # CONVOLUTION LAYER 2
    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),


    # FLATTEN
    Flatten(),


    # FULLY CONNECTED
    Dense(128, activation='relu'),

    Dropout(0.5),


    # OUTPUT LAYER
    Dense(10, activation='softmax')

])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# TRAIN MODEL
# =========================

history = model.fit(

    x_train,
    y_train,

    epochs=30,

    batch_size=64,

    validation_data=(x_test, y_test)

)

# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(x_test, y_test)

print("\n==========================")
print("CNN Accuracy :", accuracy)
print("CNN Loss     :", loss)
print("==========================")

# =========================
# SAVE MODEL
# =========================

model.save("model.h5")

print("\nModel saved: model.h5")

# =========================
# PLOT ACCURACY
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend([
    'Train Accuracy',
    'Validation Accuracy'
])

plt.grid()

plt.show()

# =========================
# PLOT LOSS
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend([
    'Train Loss',
    'Validation Loss'
])

plt.grid()

plt.show()