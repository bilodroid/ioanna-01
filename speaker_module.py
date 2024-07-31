import os
import wave
import pyaudio
from google.cloud import texttospeech

class SpeakerModule:
    def __init__(self, credentials_path='./resources/gcp_speech_and_text_credentials.json'):
        self.tts_client = texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)
        self.tts_output_filename = "tts_output.wav"
        self.p = pyaudio.PyAudio()
        self.chunk = 1024

    def synthesize_speech(self, text):
        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = self.tts_client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )

        with open(self.tts_output_filename, 'wb') as out:
            out.write(response.audio_content)

        self.play(self.tts_output_filename)

    def play(self, filename):
        wf = wave.open(filename, 'rb')

        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

        data = wf.readframes(self.chunk)

        while data != b'':
            stream.write(data)
            data = wf.readframes(self.chunk)

        print("* done speaking")
        print("-----")

        stream.stop_stream()
        stream.close()

        wf.close()

    def delete_tts_output(self):
        try:
            if os.path.exists(self.tts_output_filename):
                os.remove(self.tts_output_filename)
                print(f"Deleted TTS output file: {self.tts_output_filename}")
            else:
                print(f"TTS output file not found: {self.tts_output_filename}")
        except Exception as e:
            print(f"Error deleting TTS output file: {str(e)}")
#
#
#
#
#
