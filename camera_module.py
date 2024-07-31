import cv2
import time
import dlib
import numpy as np
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable
from threading import Lock

class CameraModule:
    def __init__(self, credentials_path='./resources/gcp_vision_credentials.json'):
        self.cap = None
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('./resources/dlib_face_recognition_resnet_model_v1.dat')
        self.client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
        self.face_detected = False
        self.last_emotion_detection_time = 0
        self.frame_lock = Lock()
        self.current_frame = None

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("cannot open webcam!")

    def detect_face(self):
        frame = self.get_current_frame()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            self.face_detected = len(faces) > 0
        return self.face_detected

    def perform_face_encoding(self, frame):
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            shape = self.sp(gray, faces[0])
            face_descriptor = self.facerec.compute_face_descriptor(frame, shape)
            return np.array(face_descriptor).tolist()
        return None

    def get_current_frame(self):
        with self.frame_lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            return self.current_frame

    def detect_emotion(self, frame):
        current_time = time.time()
        if current_time - self.last_emotion_detection_time >= 0.5:
            success, encoded_image = cv2.imencode('.jpg', frame)
            if success:
                image_content = encoded_image.tobytes()
                image = vision.Image(content=image_content)

                retries = 3
                for attempt in range(retries):
                    try:
                        response = self.client.face_detection(image=image)
                        face_annotations = response.face_annotations
                        if face_annotations:
                            face = face_annotations[0]
                            emotions = {
                                'anger': face.anger_likelihood,
                                'joy': face.joy_likelihood,
                                'sorrow': face.sorrow_likelihood,
                                'surprise': face.surprise_likelihood,
                            }
                            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
                            likelihood_values = {
                                'UNKNOWN': 0,
                                'VERY_UNLIKELY': 1,
                                'UNLIKELY': 2,
                                'POSSIBLE': 3,
                                'LIKELY': 4,
                                'VERY_LIKELY': 5,
                            }
                            emotion_texts = [f"{emotion}: {likelihood_values[likelihood_name[emotions[emotion]]]}" for emotion in emotions]
                            self.last_emotion_detection_time = current_time
                            return emotion_texts
                        break
                    except ServiceUnavailable as e:
                        print(f"service unavailable, retrying ({attempt+1}/{retries})!")
                        time.sleep(1)
                    except Exception as e:
                        print(f"an error occurred: {str(e)}!")
                        break
        return None

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
#
#
#
#
#
