import pyaudio
import numpy as np
import socketio
import struct

# WebSocket server URL (replace with your actual WebSocket server URL)
WEBSOCKET_SERVER = 'http://192.168.190.65:6000/socket.io/websocket'

# Audio parameterso
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK_SIZE = 1024  # Number of frames per buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

# Define audio stream callback
def audio_stream_callback(in_data, frame_count, time_info, status):
    # Convert PCM audio data to bytes
    audio_bytes = struct.pack(f'{frame_count}h', *np.frombuffer(in_data, dtype=np.int16))
    sio.emit('audio_data', audio_bytes)
    return in_data, pyaudio.paContinue

# Open microphone stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=audio_stream_callback)

# Connect to the WebSocket server
sio.connect(WEBSOCKET_SERVER)

print("Streaming audio. Press Ctrl+C to stop.")

try:
    # Keep the script running
    while True:
        sio.sleep(1)  # Keep the connection alive
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Close WebSocket connection
    sio.disconnect()
    # Stop microphone stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
