"""
Show raw audio and mu-law encoded samples to make input source
"""

import os

import librosa
import numpy as np

import torch as t
import torch.utils.data as data

def load_audio(path, sample_rate=16000, trim=True, trim_frame_length=2048):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)

    if trim:
        audio, _ = librosa.effects.trim(audio, frame_length=trim_frame_length)
    
    return audio

def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=np.float32)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot

def one_hot_decode(data, axis=1):
    return np.argmax(data, axis=axis)

def mu_law_encode(audio, quantization_channels=256):
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(1 + mu)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized

def mu_law_decode(output, quantization_channels=256):
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (np.power(1 + mu, np.abs(expanded)) - 1) / mu

    return waveform

class Dataset(data.Dataset):
    def __init__(self, data_dir, sample_rate=16000, in_channels=256, trim=True):
        super().__init__()

        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(data_dir))]

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])

        raw_audio = load_audio(filepath, sample_rate=self.sample_rate, trim=self.trim)

        encoded_audio = mu_law_encode(raw_audio, quantization_channels=self.in_channels)
        one_hot_encoded_audio = one_hot_encode(encoded_audio, channels=self.in_channels)

        return one_hot_encoded_audio

    def __len__(self):
        return len(self.filenames)

class DataLoader(data.DataLoader):
    def __init__(self, data_dir, receptive_fields, sample_size=0, 
                    sample_rate=16000, in_channels=256, batch_size=1, shuffle=True):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
        :param sample_size: integer. number of timesteps to train at once.
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param sample_rate: sound sampling rates
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """ 
        dataset = Dataset(data_dir, sample_rate, in_channels)
        super().__init__(dataset, batch_size, shuffle)

        if sample_size <= receptive_fields:
            raise ValueError("sample_size has to be bigger than receptive fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields

        self.collate_fn = self._collate_fn
    
    def calc_sample_size(self, audio):
        return self.sample_size if len(audio[0]) >= self.sample_size else len(audio[0])

    @staticmethod
    def _variable(data):
        tensor = t.from_numpy(data).float()

        if t.cuda.is_available():
            return t.autograd.Variable(tensor.cuda())
        else:
            return t.autograd.Variable(tensor)

    def _collate_fn(self, audio):
        audio = np.pad(audio, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        if self.sample_size:
            sample_size = self.calc_sample_size(audio)

            while sample_size > self.receptive_fields:
                inputs = audio[:, :sample_size, :]
                targets = audio[:, self.receptive_fields:sample_size, :]

                yield self._variable(inputs), self._variable(one_hot_decode(targets, 2))

                audio = audio[:, (sample_size - self.receptive_fields):, :]
                sample_size = self.calc_sample_size(audio)
        else:
            targets = audio[:, self.receptive_fields:, :]
            return self._variable(audio), self._variable(one_hot_decode(targets, 2))