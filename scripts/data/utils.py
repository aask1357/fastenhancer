from pathlib import Path
import typing as tp
import os
import random
import math
import io
import subprocess
from tqdm import tqdm
import wave

from lhotse import CutSet, Recording
from lhotse.cut import Cut
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import librosa
from scipy import signal
import soundfile as sf

from scripts.audiolib import (
    EPS, active_rms_relative, normalize_segmental_rms,
    find_rir_onset_spectral, get_rir_start_sample,
)

MAXTRIES = 50
TARGET_DB_FOR_LOADING = -25


class RandomGain:
    def __init__(self, low_db: float, high_db: float):
        self.low_db = low_db
        self.high_db = high_db
    
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(self.low_db, self.high_db)
        gain = 10**(gain_db / 20)   # dB -> amplitude# dB -> amplitude
        return wav * gain


AUDIO_EXT = [".wav", ".WAV", ".flac", ".FLAC", ".mp3"]


def is_audiofile(file: str) -> bool:
    return any(file.endswith(ext) for ext in AUDIO_EXT)


def is_directory_to_exclude(cur_dir: Path, blacklist: tp.List[str]) -> bool:
    parent_dir = cur_dir.parents
    for bl in blacklist:
        bl = Path(bl)
        if bl in parent_dir:
            return True
        if bl == cur_dir:
            return True
    return False


def is_file_to_exclude(file: Path, blacklist: tp.List[Path]) -> bool:
    return file in [Path(bl) for bl in blacklist]


class Directories:
    def __init__(
        self,
        directories_to_include: tp.List[str],
        directories_to_exclude: tp.List[str] = [],
        extension: str = "",
        mix: tp.Optional[tp.Dict[str, float]] = None,
        files_to_exclude: tp.List[Path] = [],
    ):
        self.extension = extension
        self.names_to_mix: tp.List[str] = []
        self.probabilities: tp.List[float] = []
        if mix is not None:
            for name, prob in mix.items():
                self.names_to_mix.append(name)
                self.probabilities.append(prob)
            self.names_to_mix.append("")
            self.probabilities.append(1.0 - sum(self.probabilities))
        
        self.dir_filelist: tp.Dict[str, tp.List[str]] = {}

        self.total_lengths = 0
        lengths = {}
        for directory in directories_to_include:
            file_list = []
            if directory.endswith(".tsv"):
                with open(directory, "r") as f:
                    for line in f.readlines():
                        file = line.strip().split("\t")[0]
                        if extension == "":
                            if is_audiofile(file):
                                file_list.append(file)
                        elif file.endswith(extension):
                            file_list.append(file[:-len(extension)])
                directory = Path(directory).parent
            else:
                for root, _, files in os.walk(directory, followlinks=True):
                    root = Path(root)
                    
                    if is_directory_to_exclude(root, directories_to_exclude):
                        continue
                    
                    for file in files:
                        full_path = root / Path(file)
                        
                        if full_path in files_to_exclude:
                            continue
                        
                        file = str(full_path.relative_to(directory))
                        if extension == "":
                            if is_audiofile(file):
                                file_list.append(file)
                        elif file.endswith(extension):
                            file_list.append(file[:-len(extension)])

            if len(file_list) == 0:
                raise RuntimeError(
                    f"Directory {directory} has total_lengths: {len(file_list)},"
                    " but should be > 0")

            file_list.sort()
            self.dir_filelist[directory] = file_list
            self.total_lengths += len(file_list)
            lengths[directory] = len(file_list)

        sorted_lengths = sorted(lengths.items())
        self.lengths = {directory: length for directory, length in sorted_lengths}

    def __len__(self) -> int:
        return self.total_lengths

    def choice(self) -> str:
        idx = random.randrange(self.total_lengths)
        cumsum = 0
        for directory, length in self.lengths.items():
            if idx < cumsum + length:
                file = self.dir_filelist[directory][idx - cumsum]
                full_path = os.path.join(directory, file + self.extension)
                return full_path
            cumsum += length
        raise RuntimeError(self.lengths, self.total_lengths, idx)


def numpy_to_rec(wav: np.ndarray, fs: int, rec_id: str) -> Recording:
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=fs, format='FLAC')
    audio_bytes = buffer.getvalue()
    rec = Recording.from_bytes(audio_bytes, recording_id=rec_id)
    return rec


class DirectoriesDataset:
    def __init__(self, directories, fs: int, silence_length: float,
                 activity_threshold_relative: tp.Optional[float] = None,
                 activity_threshold_absolute: tp.Optional[float] = None,
                 normalize_output: bool = True,
                 mix_random_gain: tp.List[float] = [-10.0, 10.0]):
        self.fs = fs
        self.silence_length = int(silence_length * fs)
        self.random_gain = mix_random_gain
        self.threshold: tp.Dict[str, float] = dict()
        if activity_threshold_relative is not None:
            self.threshold["relative"] = activity_threshold_relative
        if activity_threshold_absolute is not None:
            self.threshold["absolute"] = activity_threshold_absolute
        self.normalize_output = normalize_output

        # Initialize silence
        self.silence = np.zeros(self.silence_length)

        self.loaders: tp.Dict[str, Directories] = {}
        self.directories: tp.List[Directories] = []
        self.probabilities: tp.List[float] = []
        cum_prob = 0.
        for name, kwargs in directories.items():
            print(f"Loading audio from {kwargs.directories_to_include}...", end=" ")
            dirs = Directories(
                directories_to_include=kwargs.directories_to_include,
                directories_to_exclude=getattr(kwargs, "directories_to_exclude", []),
                extension=kwargs.extension,
                mix=getattr(kwargs, "mix", None),
            )
            print(f"Done. #files: {len(dirs)}")
            self.loaders[name] = dirs
            self.directories.append(dirs)
            self.probabilities.append(kwargs.probability)
            cum_prob += kwargs.probability
        assert math.isclose(cum_prob, 1.0), f"cum_prob: {cum_prob}, but should  be 1.0"

    def normalize(self, wav: np.ndarray) -> np.ndarray:
        if not self.normalize_output:
            return wav
        rms = active_rms_relative(wav, self.fs, **self.threshold)
        if rms > 0.0:
            wav = normalize_segmental_rms(wav, rms, TARGET_DB_FOR_LOADING)
        return wav

    def load_wav(
        self,
        directories: Directories,
        duration_to_load: float
    ) -> tp.Tuple[np.ndarray, str]:
        error_count = 0
        while error_count < 10:
            error_count += 1
            filepath = directories.choice()
            try:
                duration = librosa.get_duration(path=filepath)
                if duration < duration_to_load:
                    wav = librosa.load(filepath, sr=self.fs)[0]
                else:
                    offset = random.uniform(0, duration - duration_to_load)
                    wav = librosa.load(
                        filepath,
                        sr=self.fs,
                        offset=offset,
                        duration=duration_to_load
                    )[0]
                return self.normalize(wav), filepath
            except Exception as e:
                pass
        raise RuntimeError(f"10 times failed to load wav from {directories}")

    def build_audio(self, dirs: Directories, duration: float):
        '''Construct an audio signal from source files'''
        output_audio = np.zeros(0)
        remaining_length = int(duration * self.fs)

        # Iterate through multiple clips until we have a long enough signal
        tries_left = MAXTRIES
        filepath_list = []
        while remaining_length > 0 and tries_left > 0:
            tries_left -= 1

            # Read next audio file & normalize to target_db
            input_audio, filepath = self.load_wav(dirs, remaining_length / self.fs)
            filepath_list.append(filepath)

            # If current file is longer than remaining desired length, drop the last part
            if len(input_audio) > remaining_length:
                input_audio = input_audio[:remaining_length]

            # Concatenate current input audio to output audio stream
            output_audio = np.append(output_audio, input_audio)
            remaining_length -= len(input_audio)

            # Add some silence if we have not reached desired audio length
            if remaining_length > 0:
                silence_len = min(remaining_length, self.silence_length)
                output_audio = np.append(output_audio, self.silence[:silence_len])
                remaining_length -= silence_len

        return output_audio, filepath_list

    def __call__(self, duration: float) -> tp.Tuple[np.ndarray, str]:
        dirs: Directories = np.random.choice(
            self.directories,         # type: ignore
            p=self.probabilities
        )
        wav, filepath_list = self.build_audio(dirs, duration)

        if dirs.names_to_mix:
            name_to_mix = np.random.choice(dirs.names_to_mix, p=dirs.probabilities)
            if name_to_mix:
                dirs_to_mix = self.loaders[name_to_mix]
                wav_to_mix, filepath_list2 = self.build_audio(dirs_to_mix, duration)
                gain_db = np.random.uniform(*self.random_gain)
                gain = 10 ** (gain_db / 20)   # dB -> amplitude
                wav = wav + wav_to_mix * gain
                if self.normalize_output:
                    rms = active_rms_relative(wav, self.fs, **self.threshold)
                    wav = normalize_segmental_rms(wav, rms, TARGET_DB_FOR_LOADING)
                filepath_list = filepath_list + filepath_list2
        
        return wav, ",".join(filepath_list)


class ReverbDataset:
    def __init__(
        self,
        fs: int,
        hps,
    ) -> None:
        self.fs = fs
        self.dataset_dict = {}

        self.loaders: tp.List[tp.Dict[str, tp.Any]] = []
        self.probabilities: tp.List[float] = []
        cum_prob = 0.
        for name, kwargs in hps.items():
            with open(kwargs["csv"], "r") as f:
                filelists = [line.strip().split(",") for line in f.readlines()[1:]]
            loader = dict(
                base_dir=kwargs["base_dir"],
                filelists=filelists,
            )
            self.loaders.append(loader)
            self.probabilities.append(kwargs["prob"])
            cum_prob += kwargs["prob"]
        assert math.isclose(cum_prob, 1.0), f"cum_prob: {cum_prob}, but should  be 1.0"

    def __call__(self) -> tp.Tuple[np.ndarray, float, float]:
        loader = np.random.choice(self.loaders, p=self.probabilities)  # type: ignore
        path, ch, t60, *_ = random.choice(loader["filelists"])
        ch = int(ch)
        t60 = float(t60)
        full_path = os.path.join(loader["base_dir"], path)
        rir, rir_fs = librosa.load(full_path, sr=None, mono=False)
        if rir.ndim > 1:
            rir = rir[ch-1]
        if rir_fs != self.fs:
            raise ValueError(f"RIR sampling rate {rir_fs} does not match target fs {self.fs}")
        onset_sample, _ = find_rir_onset_spectral(rir, rir_fs)
        onset_heuristic = get_rir_start_sample(rir).item()
        if abs(onset_sample - onset_heuristic) > 0.001 * self.fs:
            print(
                f"Onset mismatch for file {full_path} channel {ch}: "
                f"spectral={onset_sample} vs heuristic={onset_heuristic}"
            )
        return rir, t60, onset_sample


class AECChallengeReal:
    def __init__(self, hps):
        hp = hps["farend"]["aec_challenge_real"]["ser"]
        ser_list = list(range(hp[0], hp[1] + 1, hp[2]))
        self.ser_list = [float(ser) for ser in ser_list]

        self.farend_echo_list = []
        ncc_threshold = hps["farend"]["aec_challenge_real"]["ncc_threshold"]
        for hp in hps["input"]["farend"]["aec_challenge_real"]:
            base_dir = Path(hp["base_dir"])
            with open(hp["tsv"], "r") as f:
                lines = f.readlines()[1:]
            for line in lines:
                farend, echo, ncc = line.strip().split("\t")
                ncc = float(ncc)
                if ncc >= ncc_threshold:
                    self.farend_echo_list.append(
                        (str(base_dir / farend), str(base_dir / echo))
                    )

    def __call__(self, length_target: int) -> tp.Tuple[np.ndarray, np.ndarray, float]:
        farend_path, echo_path = random.choice(self.farend_echo_list)
        farend, _ = sf.read(farend_path)
        echo, _ = sf.read(echo_path)

        if len(farend) >= length_target:
            farend = farend[:length_target]
        else:
            farend = np.pad(farend, (0, length_target - len(farend)), mode="constant")
        if len(echo) >= length_target:
            echo = echo[:length_target]
        else:
            echo = np.pad(echo, (0, length_target - len(echo)), mode="constant")

        ser = random.choice(self.ser_list)
        return farend, echo, ser


class AECChallengeSynthetic:
    def __init__(self, hps):
        hp = hps["farend"]["aec_challenge_synthetic"]["ser"]
        ser_list = list(range(hp[0], hp[1] + 1, hp[2]))
        self.ser_list = [float(ser) for ser in ser_list]

        hp = hps["input"]["farend"]["aec_challenge_synthetic"]
        self.base_dir = Path(hp["base_dir"])
        self.idx_lower = hp["id"]["lower"]
        self.idx_upper = hp["id"]["upper"]

    def __call__(self, length_target: int) -> tp.Tuple[np.ndarray, np.ndarray, float]:
        idx = random.randint(self.idx_lower, self.idx_upper)
        farend_path = self.base_dir / "farend_speech" / f"farend_speech_fileid_{idx}.wav"
        echo_path = self.base_dir / "echo_signal" / f"echo_fileid_{idx}.wav"
        farend = librosa.load(farend_path, sr=None)[0]
        echo = librosa.load(echo_path, sr=None)[0]
        assert len(farend) == len(echo), f"Length mismatch between farend and echo for idx {idx}"
        length = len(farend)
        if length >= length_target:
            farend = farend[:length_target]
            echo = echo[:length_target]
        else:
            farend = np.pad(farend, (0, length_target - length), mode="constant")
            echo = np.pad(echo, (0, length_target - length), mode="constant")

        ser = random.choice(self.ser_list)
        return farend, echo, ser


class Farend:
    def __init__(self, hps):
        self.fs = hps["fs"]
        self.prob_farend_echo = hps["farend"]["prob"]["farend_echo"]
        self.prob_farend_only = hps["farend"]["prob"]["farend_only"]
        self.prob_real = hps["farend"]["aec_challenge_real"]["prob"]
        self.prob_synthetic = hps["farend"]["aec_challenge_synthetic"]["prob"]
        assert math.isclose(
            self.prob_real + self.prob_synthetic, 1.0
        ), (
            f"prob_real: {self.prob_real}, "
            f"prob_synthetic: {self.prob_synthetic}, but should sum to 1.0"
        )

        self.dataset_real = AECChallengeReal(hps)
        self.dataset_synthetic = AECChallengeSynthetic(hps)

    def __call__(self, duration: float) -> tp.Tuple[np.ndarray, np.ndarray, float, str, bool, bool]:
        # return: farend, echo, ser, is_real, farend_exists, echo_exists
        p = random.random()
        if p < self.prob_farend_echo:
            farend_exists, echo_exists = True, True
        elif p < self.prob_farend_echo + self.prob_farend_only:
            farend_exists, echo_exists = True, False
        else:
            farend_exists, echo_exists = False, False

        p = random.random()
        length_target = int(duration * self.fs)
        if p < self.prob_real:
            farend, echo, ser = self.dataset_real(length_target)
            farend_type = "real"
        else:
            farend, echo, ser = self.dataset_synthetic(length_target)
            farend_type = "synthetic"
        return farend, echo, ser, farend_type, farend_exists, echo_exists
