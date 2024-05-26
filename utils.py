import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir, names_file):
    names_df = pd.read_csv(names_file, header=None, names=['id', 'breed'])
    name_encoder = LabelEncoder()
    names_df['breed'] = name_encoder.fit_transform(names_df['breed'])
    
    images = []
    names = []
    
    for index, row in names_df.iterrows():
        img_path = os.path.join(data_dir, row['id'] + '.jpg')
        if os.path.exists(img_path):
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            images.append(image)
            names.append(row['breed'])
    
    images = np.array(images, dtype='float32') / 255.0
    names = np.array(names)
    
    return images, names, name_encoder.classes_

def load_test_data(test_dir):
    test_images = []
    test_image_ids = []
    
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        image = load_img(img_path, target_size=(128, 128))
        image = img_to_array(image)
        test_images.append(image)
        test_image_ids.append(img_name.split('.')[0])
    
    test_images = np.array(test_images, dtype='float32') / 255.0
    
    return test_images, test_image_ids
