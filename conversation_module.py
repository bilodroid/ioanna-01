from PyQt5.QtCore import QThread, pyqtSignal
from camera_module import CameraModule
from speaker_module import SpeakerModule
from microphone_module import MicrophoneModule
from ioanna_module import Ioanna
import spacy
from pymongo import MongoClient
import numpy as np
import threading
import time
import datetime
from threading import Lock

class ThreadSafeConversationHistory:
    def __init__(self):
        self._history = []
        self._lock = Lock()

    def append(self, message):
        with self._lock:
            self._history.append(message)

    def get_history(self):
        with self._lock:
            return list(self._history)

class ConversationModule(QThread):
    face_detected_signal = pyqtSignal(bool)
    finished = pyqtSignal(bool)
    conversation_updated = pyqtSignal(list)
    new_message = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = CameraModule()
        self.speaker = SpeakerModule()
        self.microphone = MicrophoneModule()
        self.mongo_client = MongoClient("")
        self.db = self.mongo_client.ioanna_1
        self.users_collection = self.db.users
        self.nlp = spacy.load("en_core_web_sm")
        self.emotions = []
        self.emotion_detection_active = False
        self.emotion_lock = Lock()
        self.running = True
        self.run_lock = Lock()

        api_key = ""
        self.conversation_history = ThreadSafeConversationHistory()
        self.ioanna = Ioanna(api_key, follow_up_limit=1, conversation_history=self.conversation_history)

    def run(self):
        try:
            print("-----")
            self.camera.start_camera()
            current_user = self.get_current_user()

            goodbye_phrase = "goodbye"

            while self.is_running():

                question = self.ioanna.get_question(current_user)
                self.conversation_history.append({'role': 'assistant', 'content': question})
                self.new_message.emit({'role': 'assistant', 'content': question})
                self.speaker.synthesize_speech(question)

                recording_started = time.time()
                processed_emotions, sentences, transcript = self.record_audio_and_facial_emotions()
                self.conversation_history.append({'role': 'user', 'content': transcript})
                self.new_message.emit({'role': 'user', 'content': transcript})

                matched_data = self.match_facial_emotions_to_sentences(processed_emotions, sentences, recording_started)
                merged_sentences = self.merge_emotion_data(sentences, matched_data)
                self.print_merge_details(merged_sentences)

                print("-----")
                details = self.memorise_sentences(merged_sentences)

                user = self.add_to_user_memories(current_user, question, details)
                self.users_collection.update_one({'user_name': user['user_name']}, {'$set': user})

                self.speaker.delete_tts_output()
                self.microphone.delete_audio_files()
                print("-----")

                if goodbye_phrase in transcript.lower():
                    break

            self.speaker.synthesize_speech("Goodbye!")
        except Exception as e:
            print(f"An error occurred in the conversation module: {str(e)}")
        finally:
            self.cleanup()

    def is_running(self):
        with self.run_lock:
            return self.running

    def stop(self):
        with self.run_lock:
            self.running = False

    def cleanup(self):
        print("cleaning up resources.")
        self.camera.stop_camera()
        self.mongo_client.close()
        self.finished.emit(True)

    def get_current_user(self):
        print("Searching for face...")
        face_detected = False

        while not face_detected and self.is_running():
            face_detected = self.camera.detect_face()
            if face_detected:
                break
            time.sleep(1)

        if not self.is_running():
            return None

        print("-----")
        print("Face detected!")
        print("-----")
        self.face_detected_signal.emit(True)

        face_encoding = None
        while face_encoding is None and self.is_running():
            frame = self.camera.get_current_frame()
            face_encoding = self.camera.perform_face_encoding(frame)
            if face_encoding is None:
                print("unable to obtain face encoding, retrying!")
                self.speaker.synthesize_speech("i couldn't recognize your face, please try again!")
                time.sleep(1)

        if not self.is_running():
            return None

        match_found, user = self.check_face_encoding(face_encoding)

        if match_found:
            greeting = f"Welcome back!"
            print(greeting)
            self.speaker.synthesize_speech(greeting)
        else:
            greeting = "Hello there, I am Ioanna! What is your name?"
            print(greeting)
            self.speaker.synthesize_speech(greeting)

            file_name = 'output.wav'
            self.microphone.record(file_name)
            transcript = self.microphone.transcribe_and_analyze(file_name)[0]

            user_name = self.get_user_name(transcript)
            print(f"user_name: {user_name}")

            user = {
                'face_encoding': face_encoding,
                'user_name': user_name,
                'memories': []
            }

            self.add_user_to_database(user)

        return user

    def get_user_name(self, text):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def check_face_encoding(self, face_encoding):
        if face_encoding is None:
            print("Error: face encoding is None.")
            return False, None

        match_found = False
        user = None

        all_faces = self.users_collection.find()
        for stored_face in all_faces:
            stored_encoding = np.array(stored_face['face_encoding'])
            dist = np.linalg.norm(np.array(face_encoding) - stored_encoding)
            if dist < 0.6:
                match_found = True
                user = stored_face
                break

        return match_found, user

    def add_user_to_database(self, user):
        try:
            result = self.users_collection.insert_one(user)
            print(f"User added to database with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error adding user to database: {e}")

    def record_audio_and_facial_emotions(self):
        file_name = 'output.wav'
        with self.emotion_lock:
            self.emotions = []
            self.emotion_detection_active = True

        def capture_emotions():
            while self.emotion_detection_active and self.is_running():
                frame = self.camera.get_current_frame()
                if frame is not None:
                    emotions = self.camera.detect_emotion(frame)
                    if emotions:
                        with self.emotion_lock:
                            emotion_dict = {}
                            for emotion in emotions:
                                name, value = emotion.split(': ')
                                emotion_dict[name] = int(value)
                            self.emotions.append((emotion_dict, time.time()))
                time.sleep(0.1)

        emotion_thread = threading.Thread(target=capture_emotions)
        emotion_thread.start()

        if not self.microphone.is_recording_active():
            self.microphone.record(file_name)

        with self.emotion_lock:
            self.emotion_detection_active = False
        emotion_thread.join()

        transcript, sentences = self.microphone.transcribe_and_analyze(file_name)

        with self.emotion_lock:
            facial_emotions = []
            for emotion_dict, timestamp in self.emotions:
                facial_emotions.append((emotion_dict, timestamp))

        print("Emotions detected during recording:")
        for emotions, timestamp in facial_emotions:
            print(f"{emotions} at {timestamp}")
        print("-----")

        return facial_emotions, sentences, transcript

    def match_facial_emotions_to_sentences(self, facial_emotions, sentences, start_time):
        matched_data = []
        sentence_start_time = 0
        for i, sentence_data in enumerate(sentences):
            sentence, audio_segment, duration, sentiment, audio_emotion = sentence_data
            sentence_end_time = sentence_start_time + duration / 1000.0
            
            emotions_list = []
            for emotions, timestamp in facial_emotions:
                relative_timestamp = timestamp - start_time
                if sentence_start_time <= relative_timestamp <= sentence_end_time:
                    emotions_list.append((emotions, relative_timestamp))
            matched_data.append((sentence, emotions_list, sentiment, audio_emotion, sentence_start_time, sentence_end_time))

            sentence_start_time = sentence_end_time

        return matched_data

    def merge_emotion_data(self, sentences, matched_data):
        full_emotional_data = []
        for sentence_data, matched in zip(sentences, matched_data):
            merged_sentence = {
                'text': sentence_data[0],
                'audio_segment': sentence_data[1],
                'duration': sentence_data[2],
                'sentiment': sentence_data[3],
                'audio_emotion': sentence_data[4],
                'detected_emotions': matched[1],
                'start_time': matched[4],
                'end_time': matched[5]
            }
            full_emotional_data.append(merged_sentence)
        return full_emotional_data

    def print_merge_details(self, full_emotional_data):
        for i, sentence in enumerate(full_emotional_data, 1):
            print(f"Sentence {i}:")
            print(f"  Text: {sentence['text']}")
            print(f"  Sentiment: {sentence['sentiment']}")
            print(f"  Audio Emotion: {sentence['audio_emotion']}")
            print(f"  Detected Emotions: {sentence['detected_emotions']}")
            print("")

    def memorise_sentences(self, full_emotional_data):

        memories = []

        for i, sentence in enumerate(full_emotional_data, 1):
            text = sentence['text']
            sentiment = sentence['sentiment']
            audio_emotion = sentence['audio_emotion']
            detected_emotions = sentence['detected_emotions']

            if audio_emotion['emotion'] in ['neutral', 'unknown', 'other']:
                audio_emotion_confidence = 0
            else:
                audio_emotion_confidence = audio_emotion['confidence']

            if not detected_emotions:
                detected_emotions = [(0, 0)]
            
            # Calculate the importance of the sentence.
            importance = self.memory_score(sentiment['subjectivity'], audio_emotion_confidence, detected_emotions)

            # Filter based on importance value.
            if importance >= 0.6:
                filtered_emotions = [emotions for emotions, _ in detected_emotions]
                memories.append({
                    'text': text,
                    'sentiment': sentiment,
                    'audio_emotion': audio_emotion,
                    'detected_emotions': filtered_emotions
                })

                print(f"Sentence {i}:")
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Audio Emotion: {audio_emotion}")
                print(f"Detected Emotions: {filtered_emotions}")
                print(f"Importance: {importance}")
                print("")

        print("-----")
        return memories

    def add_to_user_memories(self, user, question, memories):
        question_object = self.create_memory_object(question, memories)
        
        self.users_collection.update_one(
            {'user_name': user['user_name']},
            {'$push': {'memories': question_object}}
        )

        if 'memories' not in user:
            user['memories'] = []
        user['memories'].append(question_object)

        self.conversation_updated.emit(self.conversation_history.get_history())

        return user

    def create_memory_object(self, question, memories):
        answers = [sentence['text'] for sentence in memories]
        return {
            'question': question,
            'answers': answers
        }

    def memory_score(self, subjectivity, audio_confidence, detected_emotions):

        # Define weights.
        weight_subjectivity = 0.5
        weight_audio_confidence = 0.3
        weight_emotion_intensity = 0.2

        # Normalize subjectivity.
        norm_subjectivity = subjectivity

        # Normalize audio confidence.
        norm_audio_confidence = audio_confidence

        # Calculate emotion intensity.
        try:
            if detected_emotions and isinstance(detected_emotions[0], tuple):
                total_intensity = sum(sum(emotion_dict.values()) for emotion_dict, _ in detected_emotions if isinstance(emotion_dict, dict))
            else:
                total_intensity = 0
            max_possible_intensity = len(detected_emotions) * 4
            norm_emotion_intensity = total_intensity / max_possible_intensity if max_possible_intensity > 0 else 0
        except Exception as e:
            print(f"Error calculating emotion intensity: {e}")
            norm_emotion_intensity = 0

        # Calculate importance value
        importance_value = (weight_subjectivity * norm_subjectivity +
                            weight_audio_confidence * norm_audio_confidence +
                            weight_emotion_intensity * norm_emotion_intensity)

        return importance_value
#
#
#
#
#
