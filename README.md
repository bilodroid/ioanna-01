# note: I like to keep all my projects, all my files actually, local. it helps me deal with brain clutter (im working on it)
# for now this repository exists for my application to the fatima fellowship, i hope i get the opportunity to work on more challenges

# i will always be grateful to Dr. Ioanna Giorgi for the wonderful experience of scrating the surface of my own true capabilities <3

# Ioanna

Ioanna is an interactive AI assistant that uses computer vision, speech recognition, and natural language processing to engage in human-like conversations. It recognizes faces, detects emotions, and adapts its responses based on user interactions and emotional cues.

## Features

- Face detection and recognition
- Speech-to-text and text-to-speech conversion
- Emotion analysis through facial expressions and voice tone
- Contextual conversation based on user history
- Graphical user interface with live camera feed and chat history

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Webcam
- Microphone and speakers

## Installation

1. Clone this repository
2. Install required packages
3. Set up Google Cloud credentials:
- Place your Google Cloud credentials JSON files in the `./resources/` directory:
  - `gcp_speech_and_text_credentials.json` for speech and text services
  - `gcp_vision_credentials.json` for vision services
4. Set up MongoDB:
- Update the MongoDB connection string in `conversation_module.py` with your database credentials.
5. Download required models:
- Place the following files in the `./resources/` directory:
  - `shape_predictor_68_face_landmarks.dat`
  - `dlib_face_recognition_resnet_model_v1.dat`

## Running the Application

1. Ensure your webcam is connected and functioning.
2. Run the main script:
3. The application window will appear with two tabs:
- "Camera": Shows the live webcam feed
- "Chat": Displays the conversation history
4. Interact with Ioanna using your voice. The application will transcribe your speech, process it, and provide a spoken response.

## Project Structure

- `main.py`: Entry point of the application
- `conversation_module.py`: Manages the overall conversation flow
- `camera_module.py`: Handles face detection and emotion recognition
- `microphone_module.py`: Manages audio recording and speech-to-text conversion
- `speaker_module.py`: Handles text-to-speech conversion and audio playback
- `ioanna_module.py`: Implements the AI assistant's response generation
- `user_interface.py`: Provides the graphical interface for the application

## Note

Ensure your system's microphone and speakers are working correctly for the best experience. This project uses various APIs and may incur costs if used extensively.
