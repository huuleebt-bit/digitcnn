import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

import matplotlib.pyplot as plt

# =========================
# LOAD MNIST
# =========================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =========================
# NORMALIZE
# =========================

x_train = x_train / 255.0

x_test = x_test / 255.0

# =========================
# BUILD ANN MODEL
# =========================

model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(128, activation='relu'),

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
# TRAIN MODEL
# =========================

history = model.fit(

    x_train,

    y_train,

    epochs=5,

    validation_data=(x_test, y_test)

)

# =========================
# EVALUATE
# =========================

loss, accuracy = model.evaluate(

    x_test,

    y_test

)

print("Accuracy:", accuracy)

# =========================
# PLOT ACCURACY
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ANN Accuracy')

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

plt.title('ANN Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend([

    'Train Loss',

    'Validation Loss'

])

plt.grid()

plt.show()