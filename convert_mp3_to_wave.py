import os
from pydub import AudioSegment

def convert_mp3_to_wav(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.mp3'):
            source_file_path = os.path.join(source_folder, file_name)
            target_file_path = os.path.join(target_folder, file_name.replace('.mp3', '.wav'))

            # Command to convert using ffmpeg directly before pydub
            ffmpeg_command = f"ffmpeg -y -analyzeduration 100M -probesize 100M -i \"{source_file_path}\" \"{source_file_path}.temp.wav\""
            os.system(ffmpeg_command)

            if os.path.exists(f"{source_file_path}.temp.wav"):
                # Load the temporary wav file and export it as wav
                audio = AudioSegment.from_wav(f"{source_file_path}.temp.wav")
                audio.export(target_file_path, format="wav")
                os.remove(f"{source_file_path}.temp.wav")
            else:
                print(f"Failed to convert {source_file_path}")

source_folder = 'data/Alafasy_64kbps'
target_folder = 'data_wav/Alafasy_64kbps'
convert_mp3_to_wav(source_folder, target_folder)