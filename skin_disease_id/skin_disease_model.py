import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.callbacks import EarlyStopping



class SkinDiseaseModel():

    def __init__(self,X,y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.X = X
        self.y = y

    def initialize_basic_model(self,shape):
        """
        shape needs to be in a tuple (_,_,_)
        """
        if len(shape) != 3 and type(shape) != tuple:
            print("Expected shape as a tuple with 3 values ( , , )")
            return None

        self.model = Sequential()

        # Notice this cool new layer that "pipe" your rescaling within the architecture
        self.model.add(Rescaling(1./255, input_shape=shape))

        # Lets add 3 convolution layers, with relatively large kernel size as our pictures are quite big too
        self.model.add(layers.Conv2D(16, kernel_size=10, activation='relu'))
        self.model.add(layers.MaxPooling2D(3))

        self.model.add(layers.Conv2D(32, kernel_size=8, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))

        self.model.add(layers.Conv2D(32, kernel_size=6, activation="relu"))
        self.model.add(layers.MaxPooling2D(3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(7, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    def model_fit(self,val_split=0.2,es=None):
        """
        Will accept early stopping conditions or set as a standard
        """
        if es == None:
            es = EarlyStopping(patience=3, restore_best_weights=True)
        self.model.fit(self.X,self.y,validation_split=val_split,epochs=100,batch_size=32,verbose=1, callbacks=[es])


    def model_evaluate(self,X,y):
        """
        Used to evaluate on an X and y test
        """
        eval = self.model.evaluate(X,y)
        return eval
