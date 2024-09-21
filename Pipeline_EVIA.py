## Here we Combine the 4 working Technologies (FR, SP, STT, eLLM) and decorate them with Design
import cv2
from fer import FER
import whisper
import pyaudio
import wave
import numpy as np
from langchain_ollama import OllamaLLM

# Initialize FER detector
detector = FER()

# Initialize Whisper model
whisper_model = whisper.load_model('base')

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3")

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "temp_recording.wav"

def process_frame(frame):
    # Detect emotions in the frame
    result = detector.detect_emotions(frame)
    if result:
        # Get the most prominent emotion
        emotions = result[0]['emotions']
        max_emotion = max(emotions, key=emotions.get)
        return max_emotion, emotions
    return None, {}

# Function to record audio
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press Enter to start recording...")
    input()

    # Start recording
    record_audio()

    # Capture the last frame for emotion detection
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        return

    # Process the frame for emotion detection
    emotion, emotions = process_frame(frame)

    # Transcribe the audio
    result = whisper_model.transcribe(WAVE_OUTPUT_FILENAME, fp16=False)
    transcription = result["text"]

    # Prepare input for LLM
    llm_input = f"""
    Detected Emotion:
    - Facial Expression: {emotion}
    
    Speech Input: {transcription}

    Based on this, provide a response that continues to make the person feel better.
    """

    # Get LLM response
    llm_response = llm.invoke(input=llm_input)

    # Display results
    print(f"Detected Emotion: {emotion}")
    print(f"Transcription: {transcription}")
    print(f"LLM Response: {llm_response}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()