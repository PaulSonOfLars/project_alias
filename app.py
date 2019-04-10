# coding=utf-8
import atexit
import time
from threading import Thread

from config import Config
from modules import ai, connection, globals, led, sound

# Global inits
# ====================================================
# init and setup RPI LEDs
LED = led.Pixels()
LED.off()

# Initialize the sound objects
noise = sound.AudioPlayer("data/noise.wav", -1, "noise", True, LED)
if Config.ASSISTANT.lower() == "googlehome":
    wakeup = sound.AudioPlayer("data/ok_google.wav", 0, "wakeup", False, LED)
elif Config.ASSISTANT.lower() == "alexa":
    wakeup = sound.AudioPlayer("data/alexa.wav", 0, "wakeup", False, LED)
else:
    print("invalid assistant selection")
    exit(1)


# Main thread
def main_thread():
    noise.play()  # Start noise
    connection.send_response()  # Send system info to client

    # variables to control timing between predictions
    prev_timer = 0
    interval = 5

    # Program loop start here
    # ====================================================#
    try:
        while True:
            while stream.is_active():
                time.sleep(0.01)
                LED.off()
                current_sec = time.time()

                # If the mic is triggered an spectogram is not done, make a row more.
                if globals.MIC_TRIGGER and not globals.EXAMPLE_READY:
                    sound.make_spectrogram()

                if globals.PREDICT and globals.EXAMPLE_READY and not globals.TRAIN and not globals.RESET:
                    sample = sound.get_spectrogram()
                    print("get spectogram")
                    print(globals.EXAMPLE_READY)
                    if globals.HAS_BEEN_TRAINED:  # if model has been trained then predict
                        globals.RESULT = classifier.predict(sample).item()
                        print("GLOBAL RESULT: %d" % globals.RESULT)

                    if globals.RESULT == 1:
                        noise.stop()
                        wakeup.play()
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
                        noise.play()
                        print("start noise")
                        LED.off()
                        globals.TRIGGERED = False
                        globals.PREDICT = True
                        connection.send_response()

    except (KeyboardInterrupt, SystemExit):
        exit_handler()


# Setup
# ====================================================#
globals.initialize()
audio, stream = sound.initialize()
stream.start_stream()  # start stream
classifier = ai.Classifier()  # setup keras model

# Setup and start main thread
thread = Thread(target=main_thread)
thread.daemon = True
thread.start()

print('')
print("============================================")
print("SERVER RUNNING ON: http://" + str(Config.HOST) + ":" + str(Config.PORT))
print("============================================")
print('')

# Start socket io
if __name__ == '__main__':
    connection.socketio.on_namespace(connection.SocketNamespace("/socket", classifier, stream, noise, LED))
    connection.socketio.run(connection.app, host=Config.HOST, port=Config.PORT, debug=False, log_output=False)


def exit_handler():
    LED.off()
    stream.stop_stream()
    stream.close()
    audio.terminate()


atexit.register(exit_handler)
