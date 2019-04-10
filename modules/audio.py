import os
from queue import Queue

import numpy as np
import pyaudio
import pygame.mixer
from python_speech_features import mfcc, sigproc

from config import Config
from modules import connection, globals

pygame.mixer.init()

# Audio settings
# ====================================================#
RATE = 16000
CHUNK_SAMPLES = 1024
FEED_DURATION = 1.5  # Duration in seconds
FEED_LENGTH = np.floor(RATE * FEED_DURATION / CHUNK_SAMPLES)
# WIN_LEN = 1 / (RATE / CHUNK_SAMPLES)  # IN SECONDS # unused?
FORMAT = pyaudio.paInt16
CHANNELS = 1
# DATA = np.zeros(CHUNK_SAMPLES, dtype='int16')  # unused?
print(FEED_LENGTH)
silence_threshhold = 700
q = Queue()


class Sound:
    running_spectrogram = np.empty([0, 13], dtype='int16')
    finished_spectrogram = np.empty([0, 13], dtype='int16')

    def __init__(self, LED):
        # TODO: Check if it _has_ to be sudo
        # IF yes, can we add user to audio group?
        os.system('amixer -c 1 sset Speaker {}'.format(Config.VOLUME))
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      output=False,
                                      input=True,
                                      frames_per_buffer=CHUNK_SAMPLES,
                                      stream_callback=audio_callback)
        self.LED = LED

        # Initialize the sound objects
        self.noise = AudioPlayer("data/noise.wav", -1, "noise", True, LED)
        if Config.ASSISTANT.lower() == "googlehome":
            self.wakeup = AudioPlayer("data/ok_google.wav", 0, "wakeup", False, LED)
        elif Config.ASSISTANT.lower() == "alexa":
            self.wakeup = AudioPlayer("data/alexa.wav", 0, "wakeup", False, LED)
        else:
            print("invalid assistant selection: {}".format(Config.ASSISTANT.lower()))
            exit(1)

    def start_stream(self):
        self.stream.start_stream()

    def stop_stream(self):
        self.stream.stop_stream()

    def play_noise(self):
        self.noise.play()

    def stop_noise(self):
        self.noise.stop()

    def play_wakeup(self):
        self.wakeup.play()

    def start(self):
        self.stream.start_stream()
        self.noise.play()

    def stop(self):
        self.stream.stop_stream()
        self.noise.stop()

    def is_active(self):
        return self.stream.is_active()

    def off(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def make_spectrogram(self):
        data = q.get()

        if len(self.running_spectrogram) < FEED_LENGTH:
            # preemphasis the signal to weight up high frequencies
            signal = sigproc.preemphasis(data, coeff=0.95)
            # apply mfcc on the frames
            mfcc_feat = mfcc(signal, RATE, winlen=1 / (RATE / CHUNK_SAMPLES), nfft=CHUNK_SAMPLES * 2,
                             winfunc=np.hamming)
            self.running_spectrogram = np.vstack([mfcc_feat, self.running_spectrogram])
            connection.send_spectogram(mfcc_feat, len(self.running_spectrogram))
            print(len(self.running_spectrogram))
        else:
            self.finished_spectrogram = self.running_spectrogram
            self.running_spectrogram = np.empty([0, 13], dtype='int16')
            globals.EXAMPLE_READY = True
            globals.MIC_TRIGGER = False

    def get_spectrogram(self):
        self.finished_spectrogram = np.expand_dims(self.finished_spectrogram, axis=0)
        globals.EXAMPLE_READY = False
        return self.finished_spectrogram


# Callback on mic input
def audio_callback(in_data, frame_count, time_info, flag):
    audio_data = np.frombuffer(in_data, dtype='int16')
    if np.abs(audio_data).mean() > silence_threshhold and not globals.MIC_TRIGGER:
        globals.MIC_TRIGGER = True
    if globals.MIC_TRIGGER:
        q.put(audio_data)
    return in_data, pyaudio.paContinue


# Audio player class
# ====================================================#
class AudioPlayer:
    def __init__(self, filepath, loop, name, can_play, LED):
        super(AudioPlayer, self).__init__()
        self.filepath = os.path.abspath(filepath)
        self.loop = loop
        self.name = name
        self.canPlay = can_play
        self.player = pygame.mixer.Sound(file=self.filepath)
        self.LED = LED

    def check_if_playing(self):
        while pygame.mixer.get_busy():
            pass
        self.LED.on()

    def play(self):
        print("playing " + self.name)
        self.player.play(loops=self.loop)
        if not self.loop:
            self.check_if_playing()

    def stop(self):
        print("stopping " + self.name)
        self.player.stop()
