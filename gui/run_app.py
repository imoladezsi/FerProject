
import sys
from PyQt5 import QtWidgets

from Ui import Ui
from face_detection.CVFaceCascade import CVFaceCascade
from face_detection.CVCnnFaceDetector import CVCnnFaceDetector
from face_detection.FaceDetectionAPI import FaceDetectionAPI
from facial_expression_recognition.FerAPI import FerAPI
from facial_expression_recognition.SampleModel1 import SampleModel1
from facial_expression_recognition.SampleModel2 import SampleModel2
import os

proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cascade_path_dnn = "/home/mihai/Documents/MasterPSI/MasterPSI/masterpsi/Anul2/Data_mining/Emotion_Recognition/fer_project/FER_project/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
cascade_path_viola_jones = "/home/mihai/anaconda3/envs/FER_project/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"

app = QtWidgets.QApplication(sys.argv)

fd_algs = [CVFaceCascade(cascade_path_viola_jones),
           CVCnnFaceDetector(cascade_path_dnn)]
fd_repo = FaceDetectionAPI(fd_algs)

fer_models = [SampleModel1(), SampleModel2(),]
fer_repo = FerAPI(fer_models)


window = Ui(fd_repo, fer_repo)

# Start the event loop.
app.exec_()


