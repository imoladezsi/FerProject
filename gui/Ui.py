import os

from PyQt5 import QtWidgets, uic
from os.path import expanduser
from PyQt5.QtWidgets import *
import logging

from face_detection import FaceDetectionInterface
from facial_expression_recognition.ModelInterface import  ModelInterface
from facial_expression_recognition.FerAPI import FerAPI
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time

import cv2

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Ui(QtWidgets.QMainWindow):
    def __init__(self, face_det_repo, fer_repo):
        super(Ui, self).__init__() # Call the inherited classes __init__ method

        self.directory_path = None
        self.save_to = None
        self.fd_repo = face_det_repo
        self.fer_repo: FerAPI = fer_repo
        self.fer_options = self.fer_repo.get_options()
        self.fd_options = self.fd_repo.get_options()
        self.current_fd: FaceDetectionInterface = None
        self.current_fer: ModelInterface = None
        self.fer_model_path = None
        self.test_single_image_path = None
        self.test_video_path = None

        self.window = uic.loadUi('MainWindow.ui', self)  # Load the .ui file
        self.set_up_UI()
        self.show()  # Show the GUI

    def set_up_UI(self):
        labels = self.fd_repo.get_labels()
        self.window.faceDetComboBox.addItems(labels)
        self.current_fd = self.fd_options[0]
        labels = self.fer_repo.get_labels()
        self.window.modelComboBox.addItems(labels)
        self.current_fer: ModelInterface = self.fer_options[0]

        # set default values. These should probably be elsewhere
        self.window.learningRateLineEdit.setText(str(self.current_fer.get_init_lr()))
        self.window.dropoutLineEdit.setText(str(self.current_fer.get_dropout()))
        self.window.imgDimLineEdit.setText(str(self.current_fer.get_img_dim()))
        self.window.batchSizeLineEdit.setText(str(self.current_fer.get_batch_size()))
        self.window.epochsLineEdit.setText(str(self.current_fer.get_epochs()))
        self.window.splitLineEdit.setText(str(self.current_fer.get_split()))

        # signals and slots
        self.window.faceDetComboBox.currentIndexChanged.connect(self.onFDComboBoxChanged, self.window.faceDetComboBox.currentIndex())
        self.window.modelComboBox.currentIndexChanged.connect(self.onFERComboBoxChanged, self.window.modelComboBox.currentIndex())

    def get_classes(self):
        # should not be any empty directory
        for _, dirnames, _ in os.walk(self.directory_path):
            return dirnames

    def getDirectory(self):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"), QFileDialog.ShowDirsOnly)
        self.window.dirLabel.setText(dir_)
        self.directory_path = dir_

    def getFerModel(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fer_model, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        self.window.ferModelLabel.setText(fer_model)
        self.fer_model_path = fer_model

    def getTestImageSelection(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        single_image_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        self.window.imageSelectionLabel.setText(single_image_path)
        self.test_single_image_path = single_image_path

    def getVideoSelection(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        self.window.testVideoLabel.setText(video_path)
        self.test_video_path = video_path

    def train(self):
        try:
            img_dim = int(self.window.imgDimLineEdit.text())
            dropout = float(self.window.dropoutLineEdit.text())
            init_lr = float(self.window.learningRateLineEdit.text())
            batch_size = int(self.window.batchSizeLineEdit.text())
            classes = self.get_classes()
            depth = 3
            epochs = int(self.window.epochsLineEdit.text())
            split = int(self.window.splitLineEdit.text())
            self.current_fer.set_params(img_dim, dropout, init_lr, len(classes), depth, batch_size, epochs)
            self.current_fer.initialize_model()

            t_start = time.time()

            lb, model = FerAPI.train(self.current_fer, self.current_fd, self.directory_path, img_dim, split, epochs, batch_size)

            print(time.time() - t_start, 'seconds')

            # default save to location
            if self.save_to is None:
                self.save_to = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
                if not os.path.exists(self.save_to):
                    os.mkdir(self.save_to)

            FerAPI.save_training_data(self.current_fer, lb, self.save_to, self.current_fer.get_name(), epochs)
        except Exception as e:
            logging.exception("Error")

    def estimateEmotionSingleImage(self):
        try:
            img_dim = int(self.window.imgDimLineEdit.text())

            t_start = time.time()

            image = cv2.imread(self.test_single_image_path)

            image = cv2.resize(image, (640, 490), interpolation=cv2.INTER_AREA)

            result = FerAPI.predict(self.current_fer, self.fer_model_path, self.current_fd, image, img_dim)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600, 600)
            cv2.imshow("image", result)

            print(time.time() - t_start, 'seconds')

        except Exception as e:
            logging.exception("Error")

    def estimateEmotionVideo(self):
        try:
            img_dim = int(self.window.imgDimLineEdit.text())

            t_start = time.time()

            cap = cv2.VideoCapture(self.test_video_path)

            if not cap.isOpened():
                print("Error opening video stream or file")
                return

            # rotate_code = self.check_rotation(self.test_video_path)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10,
                                  (frame_width, frame_height), True)

            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 600, 600)

            count = 0

            # Read until video is completed
            while cap.isOpened() and count < 40:
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret:
                    count = count + 1
                    print('Counter ', count)

                    # if rotate_code is not None:
                    #     frame = self.correct_rotation(frame, rotate_code)
                    frame = cv2.flip(frame,0)
                    # cv2.imshow("image", frame)
                    result = FerAPI.predict(self.current_fer, self.fer_model_path, self.current_fd, frame, img_dim)
                    # cv2.imshow("image", result)
                    out.write(result)

                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(time.time() - t_start, 'seconds')
        except Exception as e:
            logging.exception("Error")

    def check_rotation(path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)

        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotateCode = None
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

        return rotateCode

    def correct_rotation(frame, rotateCode):
        return cv2.rotate(frame, rotateCode)

    def setDirectory(self):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"), QFileDialog.ShowDirsOnly)
        self.save_to = dir_

    def onFDComboBoxChanged(self, index):
        self.current_fd = self.fd_options[index]

    def onFERComboBoxChanged(self, index):
        self.current_fer = self.fer_options[index]



