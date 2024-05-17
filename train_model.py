import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import load_data

class DogBreedTrainer:
    def __init__(self, train_dir, img_size=(224, 224), batch_size=32, epochs=15):
        self.train_dir = train_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(os.listdir(self.train_dir)), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        train_gen, val_gen, _ = load_data(train_dir=self.train_dir, img_size=self.img_size, batch_size=self.batch_size)
        self.model.fit(train_gen, epochs=self.epochs, validation_data=val_gen)
        self.model.save('dog_breed_model.h5')

if __name__ == "__main__":
    trainer = DogBreedTrainer(train_dir='data/train')
    trainer.train()
