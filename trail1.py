import os
import numpy as np
import librosa
import soundfile as sf
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
import pyaudio
import wave
from datetime import datetime

def extract_features(audio_file):
    """Extract audio features using librosa"""
    try:
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T)
        contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T)
        tonnetz = np.mean(librosa.feature.tonnetz(y=X, sr=sample_rate).T)
        
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        return features
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def predict_gender(features):
    """Placeholder gender prediction - replace with actual model"""
    # In a real app, load your trained model here
    # gender_model = pickle.load(open('gender_model.pkl', 'rb'))
    # return gender_model.predict(features.reshape(1, -1))[0]
    return np.random.choice(['Male', 'Female'])

def predict_emotion(features):
    """Placeholder emotion prediction - replace with actual model"""
    # emotion_model = pickle.load(open('emotion_model.pkl', 'rb'))
    # return emotion_model.predict(features.reshape(1, -1))[0]
    emotions = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fear', 'Disgust', 'Surprise']
    return np.random.choice(emotions)

def record_audio(filename, duration=5, sample_rate=44100, chunk_size=1024):
    """Record audio using PyAudio"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    st.info(f"Recording for {duration} seconds...")
    frames = []
    
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    st.success("Recording complete")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def main():
    st.title("🎙️ Voice Gender & Emotion Detection")
    st.markdown("""
    This app analyzes voice recordings to predict:
    - **Gender** (Male/Female)
    - **Emotion** (Happy, Sad, Angry, etc.)
    """)
    
    # Create temp directory if needed
    os.makedirs("temp", exist_ok=True)
    
    option = st.radio("Input method:", 
                     ("Record Voice", "Upload Audio File"),
                     horizontal=True)
    
    if option == "Record Voice":
        if st.button("🎤 Start Recording"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"temp/recording_{timestamp}.wav"
            record_audio(audio_file)
            
            with st.spinner("Analyzing recording..."):
                features = extract_features(audio_file)
                if features is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Gender", predict_gender(features))
                    with col2:
                        st.metric("Detected Emotion", predict_emotion(features))
                    
                    st.audio(audio_file)
    
    else:  # Upload Audio File
        uploaded_file = st.file_uploader("Choose an audio file", 
                                       type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"temp/uploaded_{timestamp}.wav"
            
            with open(audio_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Analyzing audio..."):
                features = extract_features(audio_file)
                if features is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Gender", predict_gender(features))
                    with col2:
                        st.metric("Detected Emotion", predict_emotion(features))
                    
                    st.audio(audio_file)

if __name__ == "__main__":
    main()
