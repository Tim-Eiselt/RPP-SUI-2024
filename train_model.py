import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import load_data

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    data_dir = 'data/train'
    names_file = 'data/names.txt'
    
    images, names, classes = load_data(data_dir, names_file)
    
    X_train, X_val, y_train, y_val = train_test_split(images, names, test_size=0.2, random_state=42)
    
    model = build_model((128, 128, 3), len(classes))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)
    
    model.save('dog_breed_classifier.h5')
