
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.api.models import load_model
from keras.api.preprocessing import image
import numpy as np

model = load_model('bmw_vs_mercedes.h5')

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def classify_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    if prediction > 0.5:
        print("Это Mercedes")
    else:
        print("Это BMW")


classify_image('test_img_2.jpg')
classify_image('test_img_1.jpg')
classify_image('test_img_3.jpg')