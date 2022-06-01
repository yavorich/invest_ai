import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, SpatialDropout1D


WEIGHTS_DIR = 'pretrained_new'


# --------------------------- MODEL -----------------------------
def create_model(x_shape, y_shape, conv_size=64, conv_blocks=1, kernel_size=5, dense_size=256):
    model = Sequential()
    for i in range(conv_blocks):
        if i == 0:
            model.add(Conv1D(conv_size*2**i, kernel_size, activation='relu', input_shape=x_shape, padding='same'))
        else:
            model.add(Conv1D(conv_size*2**i, kernel_size, activation='relu', padding='same'))
        model.add(Conv1D(conv_size*2**i, kernel_size, activation='relu', padding='same'))   
        model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(y_shape[0], activation='softmax'))
    return model


def save_weights(model, path, weights_dir=WEIGHTS_DIR):
    model.save_weights(os.path.join(weights_dir, path))
    print('weights saved to', os.path.join(weights_dir, path))


def load_weights(model, path, weights_dir=WEIGHTS_DIR):
    model.load_weights(os.path.join(weights_dir, path))
    print('weights loaded from', os.path.join(weights_dir, path))