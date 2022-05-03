import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import torch
import librosa as lr
import warnings

class WindowedAudioDataset(Dataset):
    def __init__(self, arousal_file, valence_file, audio_dir,
                 window_length=20.0, window_hopsize=15.0,
                 sampling_rate=44100, max_length=45.0, mfcc_features=30,
                 transform=None, target_transform=None):
        self.mfcc_features = mfcc_features
        self.audio_dir = audio_dir
        self.sr = sampling_rate
        self.max_length = max_length
        self.arousal_file = pd.read_csv(arousal_file)
        self.valence_file = pd.read_csv(valence_file)
        self.window_length = window_length
        self.annotation_start = 15.0
        self.annotation_end = 44.0
        self.feature_hop_len = 1764
        self.window_hopsize = window_hopsize

        self.mfccs = []
        self.tempograms = []
        self.energies = []

        #self.target_transform = target_transform
        #self.transform = transform

        self.mode = 'mfcc'

    def generate_x_y(self, process_view=True):
        self.file_names = []
        self.labels = []
        print('Retrieving File Names and Labels...')
        max_range = (self.annotation_end - self.window_length / 2) * 1000 + 1

        for i, row in tqdm(self.valence_file.iterrows()):
            self.file_names.append(str(int(row['song_id'])) + '.mp3')

            for label_index in range(int(self.annotation_start * 1000), int(max_range),
                                     int(self.window_hopsize * 1000)):
                label_name = 'sample_' + str(label_index) + 'ms'
                self.labels.append([row[label_name], self.arousal_file.iloc[i].loc[label_name]])

        self.labels = np.array(self.labels)
        print('Done!')

        print('Loading Files and Features...')
        warnings.filterwarnings('ignore', '.*audioread.*')
        #self.samples = []

        n_per_second = int(self.sr / self.feature_hop_len)

        self.mfccs = []
        self.tempograms = []
        self.energies = []

        res = Parallel(n_jobs=16, backend='multiprocessing')(delayed(self.load_sample)(f) for f in tqdm(self.file_names))

        range_max = int(self.annotation_end - self.window_length / 2)
        for i in tqdm(range(len(self.file_names))):
            for ii in range(int(self.annotation_start), range_max, int(self.window_hopsize)):
                start = int((ii - self.window_length / 2) * n_per_second)
                end = int((ii + self.window_length / 2) * n_per_second)
                self.mfccs.append(res[i][0][start:end])
                self.energies.append(res[i][1][start:end])
                self.tempograms.append(res[i][2][start:end])

        self.mfccs = np.array(self.mfccs)
        self.tempograms = np.array(self.tempograms)
        self.energies = np.array(self.energies)
        #self.samples = np.array(self.samples)
        print('Done!')
        del res

    def load_sample(self, file_name):
        # DEAM Dataset includes only Audio Files with 44100Hz as native sr
        sample = lr.load(self.audio_dir + file_name, sr=self.sr,
                         duration=self.max_length)[0]

        #self.samples.append(sample)
        S = lr.stft(y=sample, n_fft=4410, hop_length=self.feature_hop_len)
        energy = lr.feature.rms(S=lr.magphase(S)[0], frame_length=4410, hop_length=self.feature_hop_len)[0]

        tempogram = lr.feature.tempogram(y=sample, sr=self.sr, hop_length=self.feature_hop_len).transpose()

        mfcc = lr.feature.mfcc(S=lr.power_to_db(lr.feature.melspectrogram(S=np.square(np.abs(S)), sr=self.sr)),
                               sr=self.sr,
                               n_mfcc=self.mfcc_features, ).transpose()
        return mfcc, energy, tempogram

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'mfcc':
            return torch.Tensor(self.mfccs[idx].transpose()), torch.Tensor(self.labels[idx])
        elif self.mode == 'energy-tempo':
            return [torch.Tensor(self.energies[idx]), torch.Tensor(self.tempograms[idx])], torch.Tensor(self.labels[idx])