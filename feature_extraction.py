import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

# Define feature extraction functions with adjustable n_fft
def extract_mfcc(y, sr, n_mfcc=13, n_fft=2048):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack((mfcc, mfcc_delta, mfcc_delta2))

def extract_spectrogram(y, sr, n_fft=2048):
    spectrogram = librosa.stft(y, n_fft=n_fft)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
    return spectrogram_db

def extract_chroma(y, sr, n_fft=2048):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    return chroma

def extract_mel_spectrogram(y, sr, n_fft=2048):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram_db

def extract_zcr(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return zcr

def extract_rmse(y):
    rmse = librosa.feature.rms(y=y)
    return rmse

def extract_spectral_features(y, sr, n_fft=2048):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    return np.vstack((spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness))

def extract_tonnetz(y, sr):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return tonnetz

#---------------------------------------------------

def preprocess_audio(file_path, sample_rate=16000, n_fft=2048):
    y, sr = librosa.load(file_path, sr=sample_rate)
    
    if len(y) < n_fft:
        print(f"Warning: Skipping {file_path} due to short duration ({len(y)} samples).")
        return None
    
    mfcc = extract_mfcc(y, sr, n_fft=n_fft).flatten()
    spectrogram = extract_spectrogram(y, sr, n_fft=n_fft).flatten()
    chroma = extract_chroma(y, sr, n_fft=n_fft).flatten()
    mel_spectrogram = extract_mel_spectrogram(y, sr, n_fft=n_fft).flatten()
    zcr = extract_zcr(y).flatten()
    rmse = extract_rmse(y).flatten()
    spectral_features = extract_spectral_features(y, sr, n_fft=n_fft).flatten()
    tonnetz = extract_tonnetz(y, sr).flatten()
    
    combined_features = np.concatenate((mfcc, spectrogram, chroma, mel_spectrogram, zcr, rmse, spectral_features, tonnetz))
    return combined_features

def save_waveform_and_spectrogram(y, sr, save_path, reciter_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.title('Waveform')
    
    plt.subplot(2, 1, 2)
    spectrogram = librosa.stft(y)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{reciter_name}.png"))
    plt.close()

def load_dataset(data_dir, output_dir, n_fft=2048):
    features = []
    labels = []
    os.makedirs(output_dir, exist_ok=True)
    reciters = os.listdir(data_dir)
    
    for label, reciter in enumerate(tqdm(reciters, desc="Reciters")):
        reciter_dir = os.path.join(data_dir, reciter)
        files = os.listdir(reciter_dir)
        
        combined_y = []
        sr = None
        
        reciter_features = []
        for file in tqdm(files, desc=f"Processing {reciter}", leave=False):
            file_path = os.path.join(reciter_dir, file)
            processed_features = preprocess_audio(file_path, n_fft=n_fft)
            y, sr = librosa.load(file_path, sr=16000)
            combined_y.extend(y)
            
            if processed_features is not None:
                reciter_features.append(processed_features)
        
        if reciter_features:
            features.append(np.mean(reciter_features, axis=0))  # Average features of all audio files for the reciter
            labels.append(label)
            
            # Save combined waveform and spectrogram for the reciter
            combined_y = np.array(combined_y)
            save_waveform_and_spectrogram(combined_y, sr, output_dir, reciter)
            
    return np.array(features), np.array(labels)

# Example usage
if __name__ == "__main__":
    data_dir = input("Enter the path to your audio dataset directory: ").strip()
    output_dir = input("Enter the path to save images: ").strip()
    
    n_fft_value = 2048  # Adjust this value as needed (512, 1024, 2048, etc.)
    features, labels = load_dataset(data_dir, output_dir, n_fft=n_fft_value)

    # Save features and labels to disk
    np.save('features.npy', features)
    np.save('labels.npy', labels)

    # Now `features` contains the combined features of each reciter, and `labels` contains the corresponding labels.
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
