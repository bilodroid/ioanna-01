import io
import wave
import pyaudio
import webrtcvad
import requests
import spacy
import os
import shutil
from google.cloud import speech
from speaker_module import SpeakerModule
from textblob import TextBlob
from pydub import AudioSegment
from io import BytesIO
from threading import Lock

try:
    spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the spaCy POS tagger')
    from spacy.cli import download
    download('en_core_web_sm')

class MicrophoneModule:
    def __init__(self, credentials_path='./resources/gcp_speech_and_text_credentials.json'):
        self.chunk = 480
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.output_filename = "output.wav"
        self.p = pyaudio.PyAudio()
        self.speech_client = speech.SpeechClient.from_service_account_json(credentials_path)
        self.speaker = SpeakerModule()
        self.vad = webrtcvad.Vad(3)
        self.max_silence_duration = 2
        self._is_recording = False
        self._recording_lock = Lock()
        self.nlp = spacy.load('en_core_web_sm')

    def is_recording_active(self):
        with self._recording_lock:
            return self._is_recording

    def record(self, file_name):
        with self._recording_lock:
            if self._is_recording:
                print("Recording is already in progress")
                return

            self._is_recording = True

        try:
            stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
            print("* recording")
            frames = []
            silence_duration = 0
            max_recording_duration = 30
            total_duration = 0

            while self.is_recording_active():
                data = stream.read(self.chunk)
                frames.append(data)
                total_duration += self.chunk / self.rate

                if self.vad.is_speech(data, self.rate):
                    silence_duration = 0
                else:
                    silence_duration += self.chunk / self.rate

                if silence_duration >= self.max_silence_duration or total_duration >= max_recording_duration:
                    break

        except Exception as e:
            print(f"An error occurred during recording: {str(e)}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            
            with self._recording_lock:
                self._is_recording = False

            print("* done recording")
            print("-----")

            wf = wave.open(file_name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

        return file_name

    def transcribe_and_analyze(self, file_name):
        with io.open(file_name, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        response = self.speech_client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        print("transcript:", transcript)
        print("-----")

        sentence_analysis = self.segment_audio(file_name, transcript)

        return transcript, sentence_analysis

    def segment_audio(self, audio_file, transcript):
        audio = AudioSegment.from_wav(audio_file)

        if self.rate != audio.frame_rate:
            print(f"Warning: self.rate ({self.rate}) does not match audio frame rate ({audio.frame_rate})")
            self.rate = audio.frame_rate

        doc = self.nlp(transcript)
        sentences = [sent.text for sent in doc.sents]

        total_audio_duration = len(audio) - (self.max_silence_duration * 1000)  # in milliseconds
        total_word_count = sum(len(sentence.split()) for sentence in sentences)
        avg_word_duration = total_audio_duration / total_word_count if total_word_count else 0

        sentence_durations = [len(sentence.split()) * avg_word_duration for sentence in sentences]

        output_dir = "segments"
        os.makedirs(output_dir, exist_ok=True)

        segments = []
        start_time = 0
        sentence_analysis = []

        for sentence, duration in zip(sentences, sentence_durations):
            end_time = start_time + duration
            segment = audio[start_time:end_time]
            if len(segment) == 0:
                print(f"Warning: Empty segment for sentence: {sentence}")

            audio_emotion = self.detect_emotion_from_audio(segment)

            segments.append(segment)
            start_time = end_time

            sentiment = self.transcript_sentiment(sentence)
            sentence_analysis.append((sentence, segment, duration, sentiment, audio_emotion))

        for i, (sentence, segment, duration, sentiment, audio_emotion) in enumerate(sentence_analysis):
            try:
                segment_path = os.path.join(output_dir, f"segment_{i}.wav")
                
                segment.export(segment_path, format="wav")

                print(f"Saved audio segment {i} to {segment_path} - Emotion detected: {audio_emotion}")

            except Exception as e:
                print(f"Error saving segment {i}: {e}")
        print("-----")

        return sentence_analysis

    def transcript_sentiment(self, text):
        blob = TextBlob(text)
        textblob_sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }

        return textblob_sentiment

    def detect_emotion_from_audio(self, audio_segment, api_url='http://127.0.0.1:8000/emotion_recognition'):
        try:
            audio_bytes = BytesIO()
            audio_segment.export(audio_bytes, format='wav')
            audio_bytes.seek(0)
            files = {'audio_file': ('audio.wav', audio_bytes, 'audio/wav')}

            os.environ['no_proxy'] = '127.0.0.1,localhost'
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                result = response.json()
                return {
                    'emotion': result['emotion'],
                    'confidence': result['confidence']
                }
            else:
                return {
                    'error': f'API request failed with status code {response.status_code}',
                    'message': response.text
                }
        except Exception as e:
            return {'error': str(e)}

    def delete_audio_files(self):
        try:
            # Delete segments directory
            if os.path.exists("segments"):
                shutil.rmtree("segments")
                print("Deleted segments directory")
            else:
                print("Segments directory not found")

            # Delete output.wav file
            if os.path.exists(self.output_filename):
                os.remove(self.output_filename)
                print(f"Deleted output file: {self.output_filename}")
            else:
                print(f"Output file not found: {self.output_filename}")
        except Exception as e:
            print(f"Error deleting audio files: {str(e)}")

    def close(self):
        self.p.terminate()
#
#
#
#
#
