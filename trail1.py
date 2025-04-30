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

# Load pre-trained models (you'll need to train or obtain these)
# For this example, we'll use placeholder functions
# In a real app, you'd replace these with actual trained models

def extract_features(audio_file):
    # Extract audio features using librosa
    try:
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T)
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T)
        contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T)
        tonnetz = np.mean(librosa.feature.tonnetz(y=X, sr=sample_rate).T)
        
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except Exception as e:
        print(f"Error encountered while parsing file: {audio_file}")
        return None
    
    return features

# Placeholder functions - replace with actual model loading and prediction
def predict_gender(features):
    # This should be replaced with your actual gender prediction model
    # For demo purposes, we'll randomly return male/female
    return np.random.choice(['Male', 'Female'])

def predict_emotion(features):
    # This should be replaced with your actual emotion prediction model
    # For demo purposes, we'll randomly return an emotion
    emotions = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fear', 'Disgust', 'Surprise']
    return np.random.choice(emotions)

# Audio recording function
def record_audio(filename, duration=5, sample_rate=44100, chunk_size=1024):
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    st.write(f"Recording for {duration} seconds...")
    frames = []
    
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    st.write("Finished recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Main Streamlit app
def main():
    st.title("Voice Gender and Emotion Detection")
    st.write("This app detects gender and emotion from voice recordings")
    
    # Create a temporary directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    option = st.radio("Select input method:", ("Record Voice", "Upload Audio File"))
    
    if option == "Record Voice":
        if st.button("Start Recording"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"temp/recording_{timestamp}.wav"
            record_audio(audio_file, duration=5)
            
            # Process the recording
            features = extract_features(audio_file)
            if features is not None:
                gender = predict_gender(features)
                emotion = predict_emotion(features)
                
                st.success("Analysis Complete!")
                st.write(f"Detected Gender: **{gender}**")
                st.write(f"Detected Emotion: **{emotion}**")
                
                # Play the recorded audio
                st.audio(audio_file)
            else:
                st.error("Error processing the audio. Please try again.")
    
    else:  # Upload Audio File
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"temp/uploaded_{timestamp}.wav"
            with open(audio_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the uploaded file
            features = extract_features(audio_file)
            if features is not None:
                gender = predict_gender(features)
                emotion = predict_emotion(features)
                
                st.success("Analysis Complete!")
                st.write(f"Detected Gender: **{gender}**")
                st.write(f"Detected Emotion: **{emotion}**")
                
                # Play the uploaded audio
                st.audio(audio_file)
            else:
                st.error("Error processing the audio. Please try a different file.")

if __name__ == "__main__":
    main()
