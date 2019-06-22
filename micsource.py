import pyaudio
import struct
import numpy as np


class AudioSource:
    def __init__(self, CHUNK, RATE):
        self.CHUNK = CHUNK
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = RATE
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.CHUNK,)

    def get_data(self):
        data = self.stream.read(self.CHUNK)
        data_int = struct.unpack(str(self.CHUNK) + 'h', data)
        # print(data_int)
        return np.array(data_int)


if __name__ == "__main__":

    source = AudioSource()
    source.get_data()
