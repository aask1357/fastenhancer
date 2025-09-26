import random
import typing as tp
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils import data
import librosa
import numpy as np


Scalar = tp.Union[int, float]


def parse_snr_range(snr_range: tp.List[int]) -> tp.List[int]:
    assert len(snr_range) == 2, f"The range of SNR should be [low, high], not {snr_range}."
    assert snr_range[0] <= snr_range[-1], f"The low SNR should not larger than high SNR."

    low, high = snr_range
    snr_list = []
    for i in range(low, high + 1, 1):
        snr_list.append(i)

    return snr_list


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def generate_a_filelist_from_a_dir(
    directory: Path,
    ext_list: tp.List[str] = [".wav", ".WAV", ".flac", ".FLAC"],
) -> tp.List[str]:
    filelist = []
    for dirpath, _, files in directory.walk():
        for filename in files:
            if any(filename.endswith(ext) for ext in ext_list):
                file = (dirpath / filename).relative_to(directory)
                filelist.append(str(file))
    return filelist


class NSOnTheFlyDataset(data.Dataset):
    """
    args:
        hp: hyperparameters
        keys (List[str]): [clean|noise|rir|is_reverb]
        mode (str): train|valid|infer|pesq
        batch_size (int): used to sort rir data by length
        verbose (bool): print debug info
    """
    def __init__(self, hp, keys: tp.List[str], textprocessor=None,
                 mode: str = "train", batch_size=1, verbose=False):
        super().__init__()
        self.keys = keys
        self.sr: int = hp.sampling_rate
        silence_length = int(hp.silence_length * self.sr)
        self.silence = np.zeros(silence_length, dtype=np.float32)

        _hp = hp.snr_mixer
        self.segmental_snr: bool = _hp.segmental_snr
        self.rms_window_size = int(_hp.rms_window_size * self.sr)
        self.activity_threshold = float(_hp.activity_threshold)
        self.target_rms = 10 ** (_hp.dataloader_rms / 20)
        self.voice_contained = int(_hp.voice_contained / _hp.rms_window_size)   # samples

        _hp = hp[mode]
        self.segment_size: int = _hp.segment_size
        self.clean_dir: Path = Path(_hp.clean_dir)
        self.noise_dir: Path = Path(_hp.noise_dir)
        self.length : int = _hp["length"]

        if verbose:
            print(f"Loading clean data from {str(self.clean_dir)}...")
        self.clean_filelist = generate_a_filelist_from_a_dir(self.clean_dir)
        if verbose:
            print(f"Total {len(self.clean_filelist)} files found.")
            print(f"Loading noise data from {str(self.noise_dir)}...")
        self.noise_filelist = generate_a_filelist_from_a_dir(self.noise_dir)
        if verbose:
            print(f"Total {len(self.noise_filelist)} files found.")

        # RIR related
        self.rir_length = 0
        self.empty_rir: np.ndarray = np.array([], dtype=np.float32)
        self.rir_dir: tp.Optional[Path] = None
        self.rir_filelist: tp.List[str] = []
        if hp.reverb_prob > 0:
            self.rir_length: int = hp.rir_length
            self.empty_rir = np.zeros(self.rir_length, dtype=np.float32)
            self.empty_rir[0] = 1.0
            self.rir_dir = Path(_hp.rir_dir)
            self.rir_filelist = generate_a_filelist_from_a_dir(self.rir_dir)

        assert 0 <= hp.reverb_prob <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_prob: float = hp.reverb_prob

        if dist.is_initialized():
            # shuffle noise & rir so that each process will not load the same file
            rng = random.Random(dist.get_rank())
            rng.shuffle(self.noise_filelist)
            rng.shuffle(self.rir_filelist)

    def __len__(self):
        return self.length

    def rms(self, wav: np.ndarray) -> float:
        if not self.segmental_snr:
            return np.sqrt(np.square(wav).mean()).item()

        # calculate rms for active segments only.
        wav_len = len(wav) // self.rms_window_size * self.rms_window_size
        wav = wav[:wav_len].reshape((-1, self.rms_window_size))

        seg_rms = np.sqrt(np.square(wav).mean(1))       # [T']
        active = seg_rms > self.activity_threshold      # [T']
        if active.sum() < self.voice_contained:
            return float("inf")     # no active segment -> return inf -> zero-out @ self.normalize
        active_seg_rms = (seg_rms * active).sum() / active.sum()
        return active_seg_rms.item()

    def normalize(self, wav: np.ndarray, rms: float) -> np.ndarray:
        return wav * (self.target_rms / (rms + 1e-12))

    def gen_audio(
        self,
        base_dir: Path,
        filelists: tp.List[str]
    ) -> tp.Tuple[np.ndarray, tp.List[str]]:
        audio_list = []
        filenames =[]
        remaining_length = self.segment_size

        while remaining_length > 0:
            file = random.choice(filelists)
            full_path = str(base_dir / file)
            audio = librosa.load(full_path, sr=self.sr)[0]
            filenames.append(file)

            audio_len = len(audio)
            if remaining_length > audio_len:
                audio_rms = self.rms(audio)
                audio = self.normalize(audio, audio_rms)
                remaining_length -= audio_len
                silence_len = min(remaining_length, len(self.silence))
                audio_list.extend([audio, self.silence[:silence_len]])
                remaining_length -= silence_len
            else:
                # 0 <= start_idx <= audio_len - remaining_length
                start_idx = random.randint(0, audio_len - remaining_length)
                audio = audio[start_idx:start_idx + remaining_length]
                audio_rms = self.rms(audio)
                audio = self.normalize(audio, audio_rms)
                audio_list.append(audio)
                remaining_length = 0

        return np.concatenate(audio_list), filenames

    def shuffle(self, seed: int) -> None:
        random.Random(seed).shuffle(self.clean_filelist)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        data = {}

        if "clean" in self.keys:
            data["clean"], clean_filenames = self.gen_audio(self.clean_dir, self.clean_filelist)

        if "noise" in self.keys:
            data["noise"], noise_filenames = self.gen_audio(self.noise_dir, self.noise_filelist)

        if "rir" in self.keys:
            use_reverb = bool(np.random.random(1) < self.reverb_prob)
            if use_reverb:
                file = random.choice(self.rir_filelist)
                full_path = self.rir_dir / file
                rir = librosa.load(full_path, sr=self.sr)[0]
                rir_len = len(rir)
                if rir_len < self.rir_length:
                    rir = np.pad(rir, (0, self.rir_length - rir_len))
                elif rir_len > self.rir_length:
                    raise RuntimeError(f"len(rir) {rir_len} > hp.rir_length {self.rir_length}")
            else:
                rir = self.empty_rir
            data["rir"] = rir

            if "is_reverb" in self.keys:
                data["is_reverb"] = use_reverb

        if "filename" in self.keys:
            data["filename"] = dict(clean=clean_filenames, noise=noise_filenames)

        return data


class SNRMixer(torch.nn.Module):
    """Mix clean & noise with a given SNR parameters.
    args:
        segmental_snr: If True, SNR is calculated using active segments only.
                     If False, SNR is calculated using the whole audio.
        snr_range: [snr_low, snr_high] in dB.
        output_dbFS_range: [dbFS_low, dbFS_high] in dB, where dbFS = max(abs(out)).
        sr: sampling rate
        energy_threshold: threshold (dB) below which a segment is considered silent.
        window_size: window size (sec) of a segment. Used only when segmental_snr = True.
        clipping_threshold: threshold for clipping.
    """
    def __init__(
        self,
        sr: int,
        segmental_snr: bool = True,
        activity_threshold: Scalar = -50,
        rms_window_size: float = 0.1,
        dataloader_rms: int = -25,
        snr_range: tp.List[int] = [-5, 20],
        noisy_rms_range: tp.List[int] = [-35, -15],
        voice_contained: float = 0.3,
        clipping_threshold: float = 1.0 - torch.finfo(torch.float32).eps,
    ):
        super().__init__()
        self.segmental_snr = segmental_snr
        self.snr_range = list(range(*snr_range))
        self.noisy_rms_range = list(range(*noisy_rms_range))
        self.sr = sr
        self.activity_threshold = 10 ** (activity_threshold / 20)   # linear scale
        self.window_size = int(sr * rms_window_size)                # samples
        self.clipping_threshold = clipping_threshold
        self.rms_dataloader = 10 ** (dataloader_rms / 20)           # linear scale
        self.voice_contained = int(voice_contained / rms_window_size)   # samples

    def scaling_while_avoiding_clipping(
        self,
        scale: Tensor,
        clean: Tensor,
        noise: Tensor,
        noisy: Tensor
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        max_abs = torch.cat(
            [x.abs().max(1, keepdim=True)[0] for x in (clean, noise, noisy)],
            dim=1
        ).max(dim=1, keepdim=True)[0]   # [B, 1]
        scale = scale.clamp(max=self.clipping_threshold / max_abs)
        clean *= scale
        noise *= scale
        noisy *= scale
        return clean, noise, noisy

    def active_rms(self, wav: Tensor) -> tp.Tuple[Tensor, Tensor]:
        batch_size = wav.size(0)
        wav_len = wav.size(1) // self.window_size * self.window_size
        wav = wav[:, :wav_len].reshape((batch_size, -1, self.window_size))

        rms = wav.square().mean(2).sqrt()       # [B, T']
        active = rms > self.activity_threshold  # [B, T']
        num_active = active.sum(1)              # [B]
        active_rms = (active * rms).sum(1) / num_active.clamp(min=1e-5)   # [B]
        mask = num_active >= self.voice_contained
        return active_rms.view(-1, 1), mask.view(-1, 1)  # [B, 1]

    def segmental_mix(
        self,
        clean: Tensor,
        noise: Tensor,
        snr: int,
        rms_target: int
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """ clean, noise: [B, T]
        Mix clean & noise with a given SNR.
        We assume that clean and noise is already normalized to
        self.rms_dataloader during dataset loading.
        However, clean may be convolved with RIR, so the rms may be
        different to self.rms_dataloader.

        If clean contains active segments less than self.voice_contained, 
        then we will assume there's no voice, and return clean=0 & noisy=noise
        """
        rms_clean, mask = self.active_rms(clean)    # [B, 1]
        clean = clean * mask    # [B, T]: zero-out non-active utterances
        rms_noise = self.rms_dataloader
        scale = rms_clean / rms_noise * 10 ** (-snr / 20)   # [B, 1]
        noise = torch.where(mask, noise * scale, noise)     # [B, T]
        noisy = clean + noise

        # Normalize to rms_target
        rms_noisy = noisy.square().mean(1, keepdim=True).sqrt()     # [B, 1]
        rms_noisy = rms_noisy.clamp(min=self.activity_threshold)
        scale = 10 ** (rms_target / 20) / rms_noisy   # [B, 1]
        clean, noise, noisy = self.scaling_while_avoiding_clipping(scale, clean, noise, noisy)

        return clean, noise, noisy

    def mix(
        self,
        clean: Tensor,
        noise: Tensor,
        snr: int,
        rms_target: int
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        rms_clean = clean.square().mean(1, keepdim=True).sqrt()
        rms_noise = self.rms_dataloader
        scale = rms_clean / rms_noise * 10 ** (-snr / 20)   # [B, 1]
        noisy = clean + noise * scale

        # Normalize to rms_target
        rms_noisy = noisy.square().mean(1, keepdim=True).sqrt()     # [B, 1]
        rms_noisy = rms_noisy.clamp(min=self.activity_threshold)
        scale = 10 ** (rms_target / 20) / rms_noisy   # [B, 1]
        clean, noise, noisy = self.scaling_while_avoiding_clipping(scale, clean, noise, noisy)

        return clean, noise, noisy

    def forward(
        self,
        clean: Tensor,
        noise: Tensor,
        rir: tp.Optional[Tensor] = None
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        '''input:
           clean, noise: [B, T] / rir: [B, T_rir]
        output:
            clean, noise, noisy: [B, 1, T]'''
        if rir is not None:
            B = clean.size(0)
            T_rir = rir.size(1)
            clean = F.pad(clean, (T_rir-1, 0))
            clean = F.conv_transpose1d(
                clean.unsqueeze(0), rir.unsqueeze(1), bias=None,
                groups=B, padding=T_rir-1)   # [1, B, T]
            clean = clean.squeeze(0)         # [B, T]

        snr = random.choice(self.snr_range)
        rms_noisy = random.choice(self.noisy_rms_range)
        if self.segmental_snr:
            return self.segmental_mix(clean, noise, snr, rms_noisy)
        else:
            return self.mix(clean, noise, snr, rms_noisy)


if __name__ == "__main__":
    from utils import HParams
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    hp = HParams(
        dataset='NSOnTheFlyDataset',
        train=HParams(
            clean_dir='/home/shahn/Datasets/DNS-Challenge4/datasets_24khz/clean',
            noise_dir='/home/shahn/Datasets/DNS-Challenge4/datasets_24khz/noise',
            segment_size=48000,
            length=16384,
        ),
        sampling_rate=24000,
        reverb_prob=0.0,
        rir_length=1,
        silence_length=0.2,
        snr_mixer=HParams(
            segmental_snr=True,
            rms_window_size=0.1,
            activity_threshold=-60,
            dataloader_rms=-25,
            snr_range=[-5, 20],
            noisy_rms_range=[-35, -15],
            voice_contained=0.3,
        ),
    )
    dataset = NSOnTheFlyDataset(hp, keys=['clean', 'noise'], verbose=True)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
    print(len(dataset), len(dataloader))
    mixer = SNRMixer(sr=hp["sampling_rate"], **hp["snr_mixer"])
    for i, batch in enumerate(tqdm(dataloader)):
        clean = batch['clean']
        noise = batch['noise']
        rir = batch.get('rir', None)
        clean, noise, noisy = mixer(clean, noise, rir)
        print(clean.shape, noise.shape, noisy.shape)
        if i == 1:
            break
