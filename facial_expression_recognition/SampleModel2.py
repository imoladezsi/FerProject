from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers import Concatenate
from keras.models import *

from facial_expression_recognition.ModelInterface import ModelInterface


class SampleModel2(ModelInterface):

    def get_history(self):
        return self._model.history

    def save(self, path):
        return self._model.save(path)

    def load_model(self, model_path):
        return load_model(model_path)

    def fit(self, train_x, train_y, test_x, test_y, epochs, batch_size):
        return self._model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs,
                               batch_size=batch_size)

    def save_weights(self, path):
        self._model.save_weights(path)

    def get_name(self):
        return self.__class__.__name__

    def __init__(self):
        super().__init__()

    def initialize_model(self):
        self._model = self.create_model()

    def create_model(self):

        padding = "valid"
        input_shape = (self._img_dim, self._img_dim, self._depth)
        input_layer = Input(input_shape)

        conv1 = Conv2D(64, 7, strides=2, padding=padding)(input_layer)
        conv1 = Activation("relu")(conv1)
        pooling1 = MaxPooling2D(3, strides=2, padding=padding)(conv1)
        # lrn1 = LRN2D()(pooling1)
        lrn1 = BatchNormalization()(pooling1)

        conv2a = Conv2D(96, 1,padding=padding)(lrn1)
        conv2a = Activation("relu")(conv2a)
        pool2a = MaxPooling2D(3, strides=1, padding=padding)(conv2a)
        # pool2a = Dropout(Constants.DROPOUT)(pool2a)
        conv2b = Conv2D(208, 3, padding=padding)(conv2a)
        conv2b = Activation("relu")(conv2b)
        conv2c = Conv2D(64, 1, padding=padding)(pool2a)
        conv2c = Activation("relu")(conv2c)
        concat2 = Concatenate(axis=-1)([conv2b, conv2c])

        pool2b = MaxPooling2D(pool_size=3, strides=1, padding=padding)(concat2)
        # pool2b = Dropout(Constants.DROPOUT)(pool2b)


        conv3a = Conv2D(96, 1, activation="relu", padding=padding)(pool2b)
        pool3a = MaxPooling2D(pool_size=3, strides = 1, padding=padding)(pool2b)
        # pool3a = Dropout(Constants.DROPOUT)(pool3a) #was commented

        conv3b = Conv2D(208, (3, 3), activation="relu", padding=padding)(conv3a)
        conv3c = Conv2D(64, (1, 1), activation="relu", padding=padding)(pool3a)

        concat3 = Concatenate(axis=-1)([conv3b, conv3c])

        pool3b = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding=padding)(concat3)

        flat = Flatten()(pool3b)
        flat = Dropout(self._dropout)(flat)
        output = Dense(self._nr_classes, activation = "softmax")(flat)

        model = Model(inputs=input_layer, outputs=output)
        model.summary()
        '''
        adam = optimizers.Adam(lr=Constants.INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=['accuracy']
                      )
        '''
        opt = SGD(lr=self._init_lr)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        return model
