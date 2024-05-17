import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import load_data

class DogBreedTester:
    def __init__(self, test_dir, model_path='dog_breed_model.h5', img_size=(224, 224), batch_size=32):
        self.test_dir = test_dir
        self.model_path = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = load_model(model_path)

    def test(self):
        _, _, test_gen = load_data(test_dir=self.test_dir, img_size=self.img_size, batch_size=self.batch_size)
        loss, accuracy = self.model.evaluate(test_gen)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f'Predicted classes: {predicted_classes}')

if __name__ == "__main__":
    tester = DogBreedTester(test_dir='data/test')
    tester.test()
