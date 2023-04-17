from torch.utils.data import Dataset, DataLoader
import librosa
import torch
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, n_mels=128):
        self.file_paths = file_paths
        self.labels = labels
        self.n_mels = n_mels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.file_paths[idx], sr=None, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=audio, n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std() 
        log_mel_spec = torch.tensor(log_mel_spec).permute(1, 0)
        label = self.labels[idx]
        return log_mel_spec, label