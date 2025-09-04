import os
import random
from typing import List, Dict, Optional, Tuple, Any
import tarfile
import io
from pathlib import Path
import wave

import torch
import numpy as np
import librosa
from tqdm import tqdm


class TensorDict(dict):
    '''To set pin_memory=True in DataLoader, need to use this class instead of dict.'''
    def pin_memory(self):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.pin_memory()


def collate(list_of_dicts: List[Dict[str, Any]]) -> TensorDict:
    """Pad the length of tensors to the maximum length."""
    data = TensorDict()
    batch_size = len(list_of_dicts)
    keys = list_of_dicts[0].keys()

    for key in keys:
        if key == "filename":
            data["filename"] = [x["filename"] for x in list_of_dicts]
            continue
        elif key.endswith("_len"):
            data[key] = torch.LongTensor([x[key] for x in list_of_dicts])
            continue
        elif key == "transcript":
            data["transcript"] = [x["transcript"] for x in list_of_dicts]
            continue
        max_len = max([x[key].size(-1) for x in list_of_dicts])
        tensor = torch.zeros(batch_size, *[x for x in list_of_dicts[0][key].shape[:-1]], max_len, dtype=list_of_dicts[0][key].dtype)
        for i in range(batch_size):
            value = list_of_dicts[i][key]
            tensor[i, ..., :value.size(-1)] = value
        data[key] = tensor
    return data


def get_snr(metadata_path: str) -> Dict[str, float]:
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    data: Dict[str, int] = dict()
    for line in lines:
        data = line.strip().split(" ")
        name = data[0]
        snr = float(data[2])
        dict[name] = snr
    return data


class VBDDataset(torch.utils.data.Dataset):
    def __init__(self, hp, keys=None, textprocessor=None, mode="train",
                 batch_size=1, verbose=False):
        super().__init__()
        if keys is None:
            self.keys = ["clean", "noisy"]
        else:
            self.keys: List[str] = keys
        self.sampling_rate = hp.sampling_rate

        self.clean_dir = hp[mode].clean_dir
        self.noisy_dir = hp[mode].noisy_dir
        self.segment_size = getattr(hp[mode], "segment_size", None)
        if mode == "infer":
            self.files = list(hp["infer"]["files"])
            self.segment_size = None
        else:
            files = [x[:-4] for x in os.listdir(self.clean_dir) if x.endswith(".wav")]
            self.files = sorted(files)
        if mode == "pesq":
            self.segment_size = None
        self.files_sorted = self.files.copy()    # deepcopy

        self.snr = dict()
        if "snr" in self.keys:
            # Currently, dataset has insufficient metadata. Don't use this.
            self.snr = get_snr(hp[mode]["metadata"])
        
        self.transcript = dict()
        if "transcript" in self.keys:
            with open(hp[mode]["transcript_dir"], "r") as f:
                lines = f.readlines()
            for line in lines:
                data = line.strip().split("|")
                filename = data[0]
                transcript = data[1]
                self.transcript[filename] = transcript

        if self.segment_size is None:
            self.batch_size = batch_size
            # Extract length of each audio
            wav_len = []
            for i in tqdm(
                range(len(self.files_sorted)),
                desc=f"Filtering {mode} dataset",
                dynamic_ncols=True,
                leave=False,
                disable=(not verbose),
            ):
                wav_length = self.get_wav_length(i)
                wav_len.append(wav_length)

            # Group audios with similar lengths in a same batch
            idx_ascending = np.array(wav_len).argsort()
            self.files_sorted = np.array(self.files_sorted)[idx_ascending]
            self.files = np.copy(self.files_sorted)

    def get_wav_length(self, idx: int) -> float:
        with wave.open(os.path.join(self.clean_dir, f"{self.files[idx]}.wav")) as f:
            return f.getnframes() / f.getframerate()

    def shuffle(self, seed: int):
        """ Shuffle the dataset with the given seed.
        Without `files_sorted.copy()`, the order will be different
        depending on the number of shuffles.
        This means that if we stop training at epoch 30 and restart later,
        the dataset order will be different from training from epoch 0
        and reaching at epoch 30."""
        if self.segment_size is None:
            rng = np.random.default_rng(seed)   # deterministic random number generator
            bs = self.batch_size
            len_ = len(self) // bs
            idx_random = np.arange(len_)
            rng.shuffle(idx_random)
            self.files[:len_ * bs] = \
                self.files_sorted[:len_ * bs].reshape((len_, bs))[idx_random, :].reshape(-1)
        else:
            self.files = self.files_sorted.copy()
            random.seed(seed)
            random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        data = {}
        x = None

        if "clean" in self.keys:
            x, sr = librosa.load(
                os.path.join(self.clean_dir, f"{filename}.wav"),
                sr=None
            )
            assert sr == self.sampling_rate
            data["clean"] = torch.from_numpy(x)

        if "noisy" in self.keys:
            x, sr = librosa.load(
                os.path.join(self.noisy_dir, f"{filename}.wav"),
                sr=None
            )
            assert sr == self.sampling_rate
            data["noisy"] = torch.from_numpy(x)
        
        if "wav_len" in self.keys:
            data["wav_len"] = len(x)

        if (self.segment_size is not None) and (x is not None):
            wav_len = len(x)
            if wav_len < self.segment_size:
                padding = self.segment_size - wav_len
                for key, value in data.items():
                    data[key] = torch.nn.functional.pad(
                        value,
                        (padding//2, padding - padding//2),
                    )
            else:
                start = random.randrange(wav_len - self.segment_size + 1)
                end = start + self.segment_size
                for key, value in data.items():
                    data[key] = value[start:end]

        if "snr" in self.keys:
            if filename in self.snr:
                snr = self.snr[filename]
            else:
                snr = float("inf")
            data["snr"] = torch.tensor([snr], dtype=torch.float32)

        if "transcript" in self.keys:
            transcript = self.transcript[filename]
            data["transcript"] = transcript
        
        if "filename" in self.keys:
            data["filename"] = filename

        return data
