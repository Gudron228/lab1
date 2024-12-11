import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.data import AUTOTUNE
import tensorflow as tf


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

def preprocess_data(ds):
    return ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE) / 255.0, y), num_parallel_calls=AUTOTUNE)

train_data = tf.keras.utils.image_dataset_from_directory(
    'data/',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    'data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = tf.keras.utils.image_dataset_from_directory(
    'test_data/',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

train_data = preprocess_data(train_data).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = preprocess_data(val_data).cache().prefetch(buffer_size=AUTOTUNE)
test_data = preprocess_data(test_data).cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


test_loss, test_acc = model.evaluate(test_data)
print(f"Точность: {test_acc * 100:.2f}%")

model.save('bmw_vs_mercedes.h5')

