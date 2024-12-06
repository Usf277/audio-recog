import random
from pydub import AudioSegment
import os

def trim_audio_files(input_folder, output_folder, target_duration):
    for subdir, _, files in os.walk(input_folder):
        # Create corresponding subfolder structure in the output directory
        relative_path = os.path.relpath(subdir, input_folder)
        output_subdir = os.path.join(output_folder, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            file_path = os.path.join(subdir, file)
            # Load audio file
            audio = AudioSegment.from_file(file_path)
            
            # Calculate duration in milliseconds
            current_duration = len(audio)
            target_duration_ms = target_duration * 1000
            
            # Check if audio is longer than target duration
            if current_duration > target_duration_ms:
                # Calculate the maximum start time that ensures a 2-second segment can be trimmed
                max_start_time = current_duration - target_duration_ms
                
                # Generate a random start time within the valid range
                start_time = random.randint(0, max_start_time)
                
                # Trim audio
                trimmed_audio = audio[start_time:start_time + target_duration_ms]
                
                # Save trimmed audio with the same folder structure
                output_file_path = os.path.join(output_subdir, file)
                trimmed_audio.export(output_file_path, format="wav")
                print(f"Trimmed {file} to {target_duration} seconds starting from {start_time / 1000} seconds.")
            else:
                print(f"{file} is already {target_duration} seconds or shorter. Skipping...")

if __name__ == "__main__":
    input_folder = "E:/Since/بحث ومقال في الحاسب مشروع التخرج/Dataset/Quran_Ayat_public/audio_data"
    output_folder = "E:/Since/بحث ومقال في الحاسب مشروع التخرج/Trimmed dataset/trimmed_audio_data"
    target_duration = 2  # Set target duration in seconds

    trim_audio_files(input_folder, output_folder, target_duration)