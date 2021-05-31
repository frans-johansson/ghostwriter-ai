import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class TextFileData(Dataset):
    def __init__(self, file_path, lower=False):
        super(TextFileData, self).__init__()

        # Read from file
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("Could not locate file")

        with open(path, 'r') as f:
            corpus = np.array(list(f.read()))

        if lower:
            corpus = np.char.lower(corpus)

        self.features = np.unique(corpus)
        self.n_features = self.features.size
        n_chars = corpus.size

        self.inp = torch.zeros(n_chars, self.n_features)
        self.inp[corpus[:, np.newaxis] == self.features[np.newaxis, :]] = 1
        self.inp.unsqueeze_(0)
        self.tgt = torch.roll(self.inp, -1, dims=1)

    def __getitem__(self, idx):
        return self.inp[:, idx], self.tgt[:, idx]

    def __len__(self):
        return self.inp.size(1)


def encode_one_hot(char, char_map):
    one_hot = torch.zeros(1, 1, len(char_map))
    one_hot[..., char == char_map] = 1
    return one_hot


def decode_indices(idxs, char_map):
    return ''.join(char_map[idxs.tolist()])
