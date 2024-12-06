import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

@socketio.on('connect', namespace='/audio')
def handle_connect():
    print('Client connected to /audio')

@socketio.on('disconnect', namespace='/audio')
def handle_disconnect():
    print('Client disconnected from /audio')

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=6000)
