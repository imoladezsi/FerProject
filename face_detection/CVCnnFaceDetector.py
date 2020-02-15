import numpy as np
import cv2
from face_detection.FaceDetectionInterface import FaceDetectionInterface
from face_detection.FaceDetectionConstants import *

import logging


class CVCnnFaceDetector(FaceDetectionInterface):
    def __init__(self, classifier_path):
        super().__init__(classifier_path)
        self._classifier = cv2.dnn.readNetFromCaffe(FaceDetectionConstants.FACE_DETECTION_PROTOTXT_PATH, classifier_path)
        self._faces = None

    def __get_faces_position(self, image):

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self._classifier.setInput(blob)
        detections = self._classifier.forward()

        maxConfidence = 0
        maxBox = []

        try:
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if (confidence > FaceDetectionConstants.CONFIDENCE_LEVEL) and (confidence > maxConfidence):
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    maxConfidence = confidence
                    maxBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            return [maxBox, maxConfidence]
        except Exception as e:
            logging.exception("Error in detecting face")


    def __crop_face(self, image, face_box):
        # image is a PIL image, for now we assume each image has a single face on it
        # needed changes: for loop here and duplicate the labels

        (startX, startY, endX, endY) = face_box.astype("int")

        face = image[startY:endY, startX:endX, :]
        # Shows the image in image viewer
        return face

    def get_name(self):
        return "DNN Face Detection"

    def get_cropped_face(self, image):
        [face_box, confidence] = self.__get_faces_position(image)
        if face_box == ():
            return
        return [self.__crop_face(image, face_box), face_box, confidence]

    def is_face(self):
        return self._faces != []
