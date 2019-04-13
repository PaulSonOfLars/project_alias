# coding=utf-8
import atexit
import time
from threading import Thread

from config import Config
from modules import ai, audio, connection, globals, led


# Main thread
def main_thread(sound: audio.Sound):
    sound.start()  # Start noise
    connection.send_response()  # Send system info to client

    # variables to control timing between predictions
    prev_timer = 0
    interval = 5

    # Program loop start here
    # ====================================================#
    try:
        while True:
            while sound.is_active():
                time.sleep(0.01)
                LED.off()
                current_sec = time.time()

                # If the mic is triggered an spectogram is not done, make a row more.
                if sound.mic_trigger and not globals.EXAMPLE_READY:
                    sound.make_spectrogram()

                if globals.PREDICT and globals.EXAMPLE_READY and not globals.TRAIN and not globals.RESET:
                    sample = sound.get_spectrogram()
                    print("get spectogram")
                    print(globals.EXAMPLE_READY)

                    if globals.HAS_BEEN_TRAINED:  # if model has been trained then predict
                        globals.RESULT = classifier.predict(sample).item()
                        print("GLOBAL RESULT: %d" % globals.RESULT)

                    if globals.RESULT == 1:
                        sound.play_wakeup()
                        globals.TRIGGERED = True
                        globals.PREDICT = False
                        prev_timer = current_sec
                        connection.send_response()

                elif globals.TRAIN:
                    classifier.train_model()
                    globals.PREDICT = True
                    globals.TRAIN = False
                    connection.send_response()  # tell client that we are done training

                else:
                    globals.RESULT = 0

                if current_sec - prev_timer > interval:
                    if globals.TRIGGERED:
                        sound.play_noise()
                        print("start noise")
                        LED.off()
                        globals.TRIGGERED = False
                        globals.PREDICT = True
                        connection.send_response()

    except (KeyboardInterrupt, SystemExit):
        exit_handler()


def exit_handler():
    LED.off()
    sound.off()


if __name__ == '__main__':
    # Setup, inits
    # ====================================================#
    # init and setup RPI LEDs
    LED = led.Pixels()
    LED.off()
    sound = audio.Sound(LED)

    classifier = ai.Classifier()  # setup keras model

    # Setup and start main thread
    thread = Thread(target=lambda: main_thread(sound))
    thread.daemon = True
    thread.start()

    print('')
    print("============================================")
    print("SERVER RUNNING ON: http://" + str(Config.HOST) + ":" + str(Config.PORT))
    print("============================================")
    print('')

    # Start socket io
    namespace = connection.SocketNamespace("/socket", classifier, sound, LED)
    connection.socketio.on_namespace(namespace)
    connection.socketio.run(connection.app, host=Config.HOST, port=Config.PORT, debug=False, log_output=False)
    atexit.register(exit_handler)
