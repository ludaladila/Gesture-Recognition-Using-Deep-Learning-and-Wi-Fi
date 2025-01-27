import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input



input_shape = (300, 234, 2)

def create_CNN1D():
    model = tf.keras.Sequential([

        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1000, 512)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),


        #layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        #layers.BatchNormalization(),
        #layers.MaxPooling1D(pool_size=2),
        #layers.Dropout(0.25),


        layers.Flatten(),


        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),


        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),


        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model

def create_CNN2D():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(100,234,2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.summary()
    for layer in model.layers:
        config = layer.get_config()
        print(f"Layer: {config['name']}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}")
    return model




def create_LSTM_model1():
    model = tf.keras.Sequential([
        # Input layer for the sequence data
        tf.keras.Input(shape=(None, 256)),  # 'None' indicates variable sequence length

        # LSTM layer with 100 hidden units
        layers.LSTM(100),

        # Dropout layer
        layers.Dropout(0.4),

        # Fully Connected Classifier layers
        layers.Dense(100, activation='relu'),
        layers.Dense(80, activation='relu'),
        layers.Dense(60, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(20, activation='relu'),

        # Softmax output layer
        layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    for layer in model.layers:
        config = layer.get_config()
        print(f"Layer: {config['name']}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}")
    return model


def create_LSTM_model():
    model = Sequential()
    model.add(LSTM(125, input_shape=(None, 468), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    for _ in range(4):
        model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model



if __name__ == "__main__":
    create_CNN2D()