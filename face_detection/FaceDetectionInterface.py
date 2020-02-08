class FaceDetectionInterface(object):
    """
    All the face detection algorithms need to implement a common interface

    """
    def __init__(self, classifier_path):
        self.__classifier = classifier_path


    def is_face(self):
        raise NotImplementedError

    def get_cropped_face(self, image):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

