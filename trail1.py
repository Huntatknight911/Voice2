import os
import numpy as np
import streamlit as st
from datetime import datetime
import soundfile as sf
from scipy.fftpack import fft
import pyaudio
import wave

# Simplified feature extraction (librosa alternative)
def extract_features(audio_file):
    try:
        data, sample_rate = sf.read(audio_file)
        if len(data.shape) > 1:  # Convert stereo to mono if needed
            data = np.mean(data, axis=1)
        fft_features = np.abs(fft(data)[:1000])
        return np.array([np.mean(fft_features), np.std(fft_features)])
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def record_audio(filename, duration=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    st.info(f"Recording for {duration} seconds...")
    
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def main():
    st.title("Voice Analysis App")
    
    os.makedirs("temp", exist_ok=True)
    
    option = st.radio("Select input method:", 
                     ("Record Voice", "Upload Audio"))
    
    if option == "Record Voice":
        if st.button("Start Recording"):
            audio_file = f"temp/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            record_audio(audio_file)
            st.audio(audio_file)
            
    else:
        uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
        if uploaded_file:
            audio_file = f"temp/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            with open(audio_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio(audio_file)

if __name__ == "__main__":
    main()
