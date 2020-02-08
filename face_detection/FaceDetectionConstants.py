import os

def get_path():
    abs_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(abs_path)
    return file_dir

class FaceDetectionConstants:
    """
    class containing the hyperparameters useful parameters that were adjusted during the
    training of the CNN models and and utilities
    """

    ############ FACE DETECTION CONSTANTS
    FACE_DETECTION_METHOD = "cnn" # can be either "cnn" or "hog"
    PROTOTXT_NAME = "deploy.prototxt.txt"
    FACE_DETECTION_PROTOTXT_PATH = os.path.join(get_path(), PROTOTXT_NAME)
    CONFIDENCE_LEVEL = 0.5


