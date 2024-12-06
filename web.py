import eventlet

eventlet.monkey_patch()

from predict import *
from flask import Flask
from flask_socketio import SocketIO
from pydub import AudioSegment
from io import BytesIO
import logging
import socket
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

audio_buffer = BytesIO()
sample_width = 2  # 2 bytes for PCM16
channels = 1
frame_rate = 44100  # Adjust based on the recording settings
confidence_threshold = 0.8
duration = 6

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return 'WebSocket Server for receiving microphone data'

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected to /audio')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected from /audio')
    global audio_buffer
    global best_prediction
    global best_confidence

    audio_buffer.flush()

    best_confidence = 0
    best_prediction = None
    

def pcm16_to_float(data):
    # Assuming 'data' is the PCM16 encoded audio bytes received via WebSocket
    # Convert PCM16 to float64 array
    print("Lenght of data" ,len(data))
    samples = np.frombuffer(data, dtype=np.int16)
    samples_float = samples.astype(np.float64) / 32768.0  # Normalize to range [-1.0, 1.0]
    return samples_float

best_prediction = None
best_confidence = 0.0

@socketio.on('audio_data')
def handle_audio_data(data):
    global audio_buffer
    global best_prediction
    global best_confidence

    audio_buffer.write(data)
    if audio_buffer.tell() >= frame_rate * sample_width * channels * duration:
        audio_buffer.seek(0)
        samples = pcm16_to_float(audio_buffer.getvalue())
        audio_buffer.flush()  # Reset buffer

        # predict
        speaker, confidence = predict_speaker_from_chunk(samples.flatten(), sr=frame_rate, augment=True)
        
        print(f"Current prediction: {speaker}, Confidence: {confidence:.2f}")

        if confidence > best_confidence:
            best_confidence = confidence
            best_prediction = speaker

        if best_confidence >= confidence_threshold:
            result = f"Final predicted Speaker: {best_prediction}, with confidence: {best_confidence:.2f}"
            print(result)

            # Reset prediction
            best_confidence = 0
            best_prediction = None

            socketio.emit("prediction", result)
            socketio.emit("disconnect", "^_^")
            

def save_wav_file(audio_segment):
    filename = f'audio_{eventlet.getcurrent().get_id()}.wav'
    with open(filename, 'wb') as f:
        audio_segment.export(f, format='wav')
    logger.info(f'Saved {filename}')

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 0))
    addr, port = s.getsockname()
    s.close()
    return port

if __name__ == '__main__':
    port = 6000
    try:
        logger.info(f'Trying to start server on port {port}')
        socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
    except OSError as e:
        if e.errno == 48:
            logger.warning(f'Port {port} is in use, finding a free port...')
            port = find_free_port()
            logger.info(f'Using port {port}')
            socketio.run(app, host='0.0.0.0', port=port)
