import json
import logging
from threading import Thread

from flask import Flask, render_template
from flask_socketio import Namespace, SocketIO

from modules import ai, audio, globals, led

# Socket I/O
# ====================================================#
app = Flask(__name__)
app.debug = False
socketio = SocketIO(app, async_mode='threading', logger=False)
socket_thread = None

logging.getLogger('werkzeug').setLevel(logging.ERROR)  # remove socket io logs


# this thread is running in the background sending data to the client when connected
def response_thread():
    print("nothing")
    # while True:


def send_spectogram(data, row):
    spec_as_list = data.tolist()  # convert from numpy to regular list
    spec_to_server = json.dumps(spec_as_list)  # convert list to json format
    socketio.emit('sound', {'spectogram': spec_to_server, 'count': row}, namespace='/socket')


def send_response():
    socketio.emit('response', {
        'result': globals.RESULT,
        'bg_examples': globals.BG_EXAMPLES,
        'tr_examples': globals.TR_EXAMPLES,
        'train_state': globals.TRAIN,
        'predict_state': globals.PREDICT,
        'reset_state': globals.RESET,
        'hasbeentrained': globals.HAS_BEEN_TRAINED,
        'triggered': globals.TRIGGERED
    }, namespace='/socket')


@app.route('/')
def index():
    print('Someone Connected!')
    global socket_thread
    if socket_thread is None:
        socket_thread = Thread(target=response_thread)
        socket_thread.start()
        send_response()
    return render_template('index.html')


class SocketNamespace(Namespace):
    def __init__(self, namespace: str, classifier: ai.Classifier, sound: audio.Sound, LED: led.Pixels):
        super().__init__(namespace)
        self.classifier = classifier  # type: ai.Classifier
        self.sound = sound  # type: audio.Sound
        self.LED = LED  # type: led.Pixels
        self.button_pressed = False

    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    def on_msg_event(self, data):
        msg = data['data']
        print(msg)
        print("----------------------")

        globals.PREDICT = False  # always stop prediction on button command

        # Add example to class 0 - Silence / background noise
        if 'class0' in msg and globals.EXAMPLE_READY:
            example = self.sound.get_spectrogram()
            self.classifier.add_example(example, 0)
            globals.BG_EXAMPLES += 1
            self.LED.listen()

        # Add example to class 1 - WakeWord
        elif 'class1' in msg and globals.EXAMPLE_READY and not Config.UPDATE_BG_DATA:
            example = self.sound.get_spectrogram()
            self.classifier.add_example(example, 1)
            globals.TR_EXAMPLES += 1
            self.LED.listen()

        # Receive train command
        elif 'train' in msg:
            globals.TRAIN = True

        # Receive reset command
        elif 'reset' in msg:
            globals.RESET = True
            send_response()  # tell client that we are reseting
            self.classifier.reset_model()
            globals.RESET = False
            globals.TR_EXAMPLES = 0

        # Toogle Alias on and off
        elif 'onoff' in msg:
            if self.sound.is_active():
                self.sound.start()
            else:
                self.sound.stop()

        # Receive is Button is pressed or released
        if 'btn_release' in msg:
            print("released")
            self.button_pressed = False
        elif 'class1' in msg or 'class0' in msg:
            self.button_pressed = True

        # Check if system is ready to predict
        if globals.TRAIN or globals.RESET or self.button_pressed or globals.TRIGGERED:
            globals.PREDICT = False
        else:
            globals.PREDICT = True

        send_response()
