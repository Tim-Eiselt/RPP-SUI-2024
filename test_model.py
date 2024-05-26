import numpy as np
import tensorflow as tf
from utils import load_test_data
import pandas as pd

if __name__ == '__main__':
    test_dir = 'data/test'
    model_path = 'dog_breed_classifier.h5'
    
    model = tf.keras.models.load_model(model_path)
    
    test_images, test_image_ids = load_test_data(test_dir)
    
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    results = pd.DataFrame({
        'id': test_image_ids,
        'predicted_breed': predicted_classes
    })
    
    results.to_csv('predictions.csv', index=False)
