import os
import pickle

from sklearn.preprocessing import LabelBinarizer
from keras.engine.saving import load_model
from keras.models import model_from_json

from facial_expression_recognition import ModelInterface
from interfaces.Repository import Repository

import numpy as np

from load_data import DataGenerator
from utils.Helpers import Helpers

import cv2

class FerAPI(Repository):

    @staticmethod
    def load_label_binarizer(model_path):
        print("[INFO] loading label binarizer...")
        lst = model_path.split(os.path.sep)
        lb = os.path.join(os.path.sep.join(lst[0:-1]), "label_binarizer_" + str(lst[-1]))
        mlb = pickle.loads(open(lb, "rb").read())
        return mlb

    @staticmethod
    def save_model(model: ModelInterface, output_path, save_as):
        print("[INFO] saving model...")
        model.save_weights(os.path.join(output_path, save_as + '.h5'))
        model.save(os.path.join(output_path, save_as))

    @staticmethod
    def save_label_binarizer(lb_object, output_path, save_as):
        print("[INFO] saving label binarizer...")
        f = open(os.path.join(output_path, "label_binarizer_" + save_as), "wb")
        f.write(pickle.dumps(lb_object))
        f.close()

    @staticmethod
    def load_model(model, model_path):
        print("[INFO] loading model..")
        return model.load_model(model_path)

    @staticmethod
    def train(model, fd_classifier, dataset_path, img_dim, split, epochs, batch_size):
        try:

            gen = DataGenerator(dataset_path)
            data, labels = gen.get_images(img_dim, fd_classifier)

            # split them manually
            split = int(len(data)*split/100)
            train_x, test_x = data[:split], data[split:]
            train_y, test_y = labels[:split], labels[split:]

            lb = LabelBinarizer()
            train_y = lb.fit_transform(train_y)
            test_y = lb.transform(test_y)

            # TODO: PNG problem
            # train model
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            test_x = np.array(test_x)
            test_y = np.array(test_y)
            return lb, model.fit(train_x, train_y, test_x, test_y, epochs, batch_size)

        except Exception as e:
            print(e)

    @staticmethod
    def save_training_data(model, lb, output_path, save_as, figure_epochs = 0):
        FerAPI.save_model(model, output_path, save_as)
        FerAPI.save_label_binarizer(lb, output_path, save_as)
        if figure_epochs != 0:
            Helpers.save_figure(model.get_history(), figure_epochs, output_path, save_as)

    @staticmethod
    def predict(model, model_path, fd_classifier, image, img_dim):

        confidence = 0
        face_image = []
        [face_image, face_box, confidence] = fd_classifier.get_cropped_face(image)
        if face_image is None:  # no face was found on this one so we skip it
            return

        # face_image = cv2.resize(face_image, (img_dim, img_dim), interpolation=cv2.INTER_AREA)
        #
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        #
        # face_image = np.expand_dims(np.expand_dims(cv2.resize(face_image, (img_dim, img_dim)), -1), 0)

        face_image = cv2.resize(face_image, (img_dim, img_dim), interpolation=cv2.INTER_AREA)

        face_image = np.expand_dims(face_image, axis=0)
        #
        # face_image = np.expand_dims(face_image, axis=3)

        # cv2.normalize(face_image, face_image, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

        # draw the bounding box of the face along with the associated
        # probability
        text = "Face detected: "
        text = text + "{:.2f}%".format(confidence * 100)

        (startX, startY, endX, endY) = face_box.astype("int")

        yFace = startY - 30
        x = startX + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (x, yFace),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # load the label binarizer
        mlb = FerAPI.load_label_binarizer(model_path)

        model = FerAPI.load_model(model, model_path)

        # lst = model_path.split(os.path.sep)
        # base_path = os.path.sep.join(lst[0:-1])
        # json_file = os.path.join(base_path, 'fer.json')
        # weights_file = os.path.join(base_path, 'fer.h5')
        #
        # json_file = open(json_file, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # model.load_weights(weights_file)

        # make a prediction on the image
        # labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        preds = model.predict(face_image)

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = mlb.classes_[i]

        emotionText = "Emotion detected: " + label

        yEmotion = startY - 15
        cv2.putText(image, emotionText, (x, yEmotion),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image

    @staticmethod
    def estimate(test_dataset_path):
        pass


