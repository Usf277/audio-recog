import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model # type: ignore
import joblib
from utils import process_single_audio_file, process_raw_audio
import sounddevice as sd
import wavio

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the trained model
model = load_model('model_with_avoid_overfitting/voice_recognition_model.h5')

# Ensure the input shape is the same as used during training
input_shape = (128, 300, 1)  # Update max_length as per your training configuration

def preprocess_audio_chunk(audio_chunk, sr=44100, augment=False):
    # Process the audio chunk to get mel-spectrogram
    mel_spectrogram = process_raw_audio(audio_chunk, sr, max_length=input_shape[1], augment=augment)
    
    # Add the channel dimension
    mel_spectrogram = mel_spectrogram[..., np.newaxis]
    
    return mel_spectrogram

def predict_speaker_from_chunk(audio_chunk, sr=44100, augment=False):
    # Preprocess the audio chunk
    mel_spectrogram_padded = preprocess_audio_chunk(audio_chunk, sr=sr, augment=augment)
    
    # Predict the speaker
    prediction = model.predict(np.array([mel_spectrogram_padded]))
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    speaker = label_encoder.inverse_transform([predicted_label])[0]
    
    return speaker, confidence

def predict_speaker_from_file(file_path, augment=False):
    # Process the audio file to get mel-spectrogram
    mel_spectrogram = process_single_audio_file(file_path, max_length=input_shape[1], augment=augment)
    
    # Add the channel dimension
    mel_spectrogram = mel_spectrogram[..., np.newaxis]
    
    # Predict the speaker
    prediction = model.predict(np.array([mel_spectrogram]))
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    speaker = label_encoder.inverse_transform([predicted_label])[0]
    
    print(f"Predicted Speaker: {speaker}, Confidence: {confidence:.2f}")

def continuous_record_and_predict(duration=2, fs=44100, confidence_threshold=0.8, output_file='recorded_audio.wav'):
    print("Recording and predicting...")
    best_prediction = None
    best_confidence = 0.0

    while True:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')  # Use 'float32' for recording
        sd.wait()
        
        # Save the recorded audio
        wavio.write(output_file, recording, fs, sampwidth=2)

        speaker, confidence = predict_speaker_from_chunk(recording.flatten(), sr=fs, augment=True)
        
        print(f"Current prediction: {speaker}, Confidence: {confidence:.2f}")

        if confidence > best_confidence:
            best_confidence = confidence
            best_prediction = speaker

        if best_confidence >= confidence_threshold:
            print(f"Final predicted Speaker: {best_prediction}, with confidence: {best_confidence:.2f}")
            break

if __name__ == "__main__":
    # Choose between recording or using an existing audio file
    choice = input("Type '1' to predict from an existing audio file, or '2' to record and predict: ").strip()

    if choice == '1':
        audio_file_path = input("Enter the path to the audio file: ").strip()
        predict_speaker_from_file(audio_file_path, augment=True)
    elif choice == '2':
        continuous_record_and_predict(duration=5, confidence_threshold=0.9)  # Adjust duration and threshold as needed
    else:
        print("Invalid choice. Please run the script again and enter '1' or '2'.")