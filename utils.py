import numpy as np
import librosa
import os
from tqdm import tqdm
import random

def pad_or_truncate(mel_spectrogram, max_length):
    if mel_spectrogram.shape[1] > max_length:
        return mel_spectrogram[:, :max_length]
    else:
        padding = np.zeros((mel_spectrogram.shape[0], max_length - mel_spectrogram.shape[1]))
        return np.hstack((mel_spectrogram, padding))

def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512, max_length=300, augment=False):
    y = y.astype(np.float32)  # Convert to floating-point

    if augment:
        # Apply data augmentation
        if random.random() < 0.5:
            y = add_background_noise(y, sr)
        if random.random() < 0.5:
            y = change_pitch(y, sr)
        if random.random() < 0.5:
            y = change_speed(y, speed_factor=random.uniform(0.8, 1.2))
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    mel_spectrogram_db = pad_or_truncate(mel_spectrogram_db, max_length)
    
    return mel_spectrogram_db

def process_audio_files(data_dir, max_length=300):
    labels = []
    mel_spectrograms = []
    
    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            for file_name in tqdm(os.listdir(person_dir), desc=f"Processing {label}"):
                file_path = os.path.join(person_dir, file_name)
                if file_path.endswith('.wav') or file_path.endswith('.mp3'):
                    mel_spectrogram = extract_mel_spectrogram_from_file(file_path, max_length=max_length, augment=True)
                    mel_spectrograms.append(mel_spectrogram)
                    labels.append(label)
                    
    return np.array(mel_spectrograms), np.array(labels)

def extract_mel_spectrogram_from_file(file_path, n_mels=128, n_fft=2048, hop_length=512, max_length=300, augment=False):
    y, sr = librosa.load(file_path, sr=None)
    return extract_mel_spectrogram(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, max_length=max_length, augment=augment)

def add_background_noise(y, sr, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def change_pitch(y, sr, n_steps=random.uniform(-2, 2)):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_speed(y, speed_factor=1.2):
    return librosa.effects.time_stretch(y, rate=speed_factor)

def process_single_audio_file(file_path, max_length=300, augment=False):
    return extract_mel_spectrogram_from_file(file_path, max_length=max_length, augment=augment)

def process_raw_audio(y, sr, max_length=300, augment=False):
    return extract_mel_spectrogram(y, sr, max_length=max_length, augment=augment)
