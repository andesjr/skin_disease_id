import tensorflow as tf
import numpy as np
from pathlib import Path
import os

IMAGE_SHAPE = (128, 128, 3)

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=IMAGE_SHAPE))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

if __name__ == '__main__':
    path = Path(__file__)
    path = path.parent.parent.absolute()
    X= np.random.random((100,128,128,3))
    y= np.random.randint(0,2,(100,))
    model = get_model()
    model.fit(X, y)
    model.save(os.path.join(path, 'saved_model/my_model'))
    print('model saved')