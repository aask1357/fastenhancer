"""
Multi-task Lhotse shard dataset with on-the-fly mixing.

ShardDataset  - DDP / multi-worker safe IterableDataset.

v3 vs. v1
──────────
1. Lazy shard discovery: shard files (manifests + tars) are discovered
   post-fork (inside __iter__), never in __init__.  Each source caches
   its file list after the first discovery so subsequent epochs pay no
   glob cost.

2. Shard-level shuffle only: at each epoch the list of shard paths is
   shuffled in-place before being passed to CutSet.from_shar().  No
   per-cut reservoir buffer (cs.shuffle(buffer_size=...)) is maintained.
   This keeps at most one tar file per field open at a time, enabling
   sequential HDD reads.

3. Auxiliary sources (noise, farend-echo, rir) still use an
   aux_buffer_size-sized reservoir for random sampling across cuts; the
   reservoir is backed by the same sequential shard reader.

DDP / multi-worker shard strategy
──────────────────────────────────
Every worker sees ALL shards, shuffled with a worker-unique seed:
    seed = global_seed + worker_id + 1000 * rank + epoch
With enough shards the probability of two workers drawing the same cut
in the same step is negligible.

Fixed steps per epoch
──────────────────────
The dataset yields batches infinitely.  The training loop counts steps
and breaks:

    for epoch in range(num_epochs):
        dataset.set_epoch(epoch)
        for step, batch in zip(range(steps_per_epoch), loader):
            ...

DynamicMixer - Pure math engine applied on GPU.  No probability parameters.
"""

from __future__ import annotations

import glob
import os
import random
import time
import typing as tp
from dataclasses import dataclass, fields as dc_fields
from typing import Any, Dict, Generator, List, Optional, Tuple
import multiprocessing as mp
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import DynamicBucketingSampler

try:
    from text.asr.korean import filter_ipa
except:
    def filter_ipa(text: str) -> str:
        raise RuntimeError("filter_ipa is not available.")
from utils import HParams
from utils.segmental_rms import segmental_rms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_KEYS: tp.FrozenSet[str] = frozenset({
    "speech",
    "speech_clean",
    "farend",
    "echo",
    "noise",
    "rir",
    "rir_onset",
    "rir_t60",
    "is_real",
    "num_samples",
    "text",
    "id_speech",
    "id_noise",
    "id_farend_echo",
    "id_rir",
})


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

@dataclass
class ShardSource:
    """One shard directory and its relative sampling weight for mux()."""
    shard_dir: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def _get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def _get_worker_info() -> Tuple[int, int]:
    info = torch.utils.data.get_worker_info()
    return (info.id, info.num_workers) if info is not None else (0, 1)


# ---------------------------------------------------------------------------
# Shard discovery (lazy, post-fork)
# ---------------------------------------------------------------------------

def _discover_shards(shard_dir: str, field_names: List[str]) -> Dict[str, List[str]]:
    """
    Returns {"cuts": [...manifest paths...], "recording": [...tar paths...], ...}.
    All lists are index-aligned: index i refers to shard i.
    Only called post-fork (inside __iter__ or lazy init paths).
    """
    cut_files = sorted(glob.glob(os.path.join(shard_dir, "cuts.*.jsonl.gz")))
    if not cut_files:
        raise FileNotFoundError(f"No cut manifests found in {shard_dir}")

    def _idx(path: str) -> int:
        return int(os.path.basename(path).split(".")[1])

    indices = [_idx(p) for p in cut_files]
    result: Dict[str, List[str]] = {"cuts": cut_files}

    for fname in field_names:
        paths: List[str] = []
        for idx in indices:
            for ext in (".tar", ".tar.gz"):
                cand = os.path.join(shard_dir, f"{fname}.{idx:06d}{ext}")
                if os.path.exists(cand):
                    paths.append(cand)
                    break
            else:
                raise FileNotFoundError(
                    f"Missing shard '{fname}.{idx:06d}.tar[.gz]' in {shard_dir}"
                )
        result[fname] = paths
    return result


def _shuffle_fields(fields: Dict[str, List[str]], seed: int) -> Dict[str, List[str]]:
    """
    Return a copy of fields with all lists permuted by the same random permutation.
    All field lists share the same index alignment, so one permutation keeps them
    consistent (cuts[i], recording[i], codec[i] still refer to the same shard).
    """
    n = len(fields["cuts"])
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    return {k: [v[i] for i in perm] for k, v in fields.items()}


# ---------------------------------------------------------------------------
# Debug helper — set env var SHARD_DEBUG=1 to activate
# ---------------------------------------------------------------------------

def _timed_gen(gen, label: str, worker_id: int = 0, threshold_s: float = 1.0):
    """
    Wrap any cut iterator to print a line whenever a single next() call takes
    longer than threshold_s seconds.  Used to pinpoint which shard source
    causes loading spikes.

    Enable with:  SHARD_DEBUG=1 python train.py ...
    """
    it = iter(gen)
    while True:
        t0 = time.perf_counter()
        try:
            cut = next(it)
        except StopIteration:
            return
        dt = time.perf_counter() - t0
        if dt > threshold_s:
            print(
                f"[SLOW CUT] {label} worker={worker_id} {dt:.2f}s  cut={getattr(cut, 'id', '?')}",
                flush=True,
            )
        yield cut


class _TimedCuts:
    """
    Re-iterable wrapper around a CutSet that times slow next() calls.
    Unlike _timed_gen (a generator), each iter() call creates a fresh
    generator so the underlying CutSet can be iterated multiple times.
    Required for speech_cuts passed to DynamicBucketingSampler, which
    calls iter() again after set_epoch().
    """
    __slots__ = ("_source", "_label", "_worker_id", "_threshold_s")

    def __init__(self, source, label: str, worker_id: int = 0, threshold_s: float = 1.0):
        self._source      = source
        self._label       = label
        self._worker_id   = worker_id
        self._threshold_s = threshold_s

    def __iter__(self):
        return _timed_gen(iter(self._source), self._label, self._worker_id, self._threshold_s)


# ---------------------------------------------------------------------------
# Speech cut source  (shard-level shuffle, one tar open at a time)
# ---------------------------------------------------------------------------

class _SpeechCutSource:
    """
    Re-iterable source of speech cuts for DynamicBucketingSampler.

    Shards are discovered lazily on the first __iter__ call (post-fork).
    Each __iter__ call shuffles the shard list using the seed set by
    set_epoch(), then opens one shard's tar files at a time for sequential
    I/O.  No per-cut shuffle buffer is maintained.

    Must be used post-fork (created inside __iter__ / _build_sampler).
    """

    __slots__ = ("_sources", "_field_names", "_shuffle", "_seed", "_discovered")

    def __init__(
        self,
        sources:     List[ShardSource],
        field_names: List[str],
        shuffle:     bool = True,
    ):
        self._sources     = sources
        self._field_names = field_names
        self._shuffle     = shuffle
        self._seed        = 0
        self._discovered: Optional[List[Tuple[Dict[str, List[str]], str, float]]] = None

    def _ensure_discovered(self) -> None:
        if self._discovered is not None:
            return
        self._discovered = []
        for src in self._sources:
            fields    = _discover_shards(src.shard_dir, self._field_names)
            id_prefix = Path(src.shard_dir).name
            self._discovered.append((fields, id_prefix, src.weight))

    def set_epoch(self, seed: int) -> None:
        """Update the shard-shuffle seed for the next __iter__ call."""
        self._seed = seed

    def __iter__(self):
        self._ensure_discovered()
        seed = self._seed

        cutsets: List[CutSet] = []
        weights: List[float]  = []

        for i, (fields, id_prefix, weight) in enumerate(self._discovered):
            effective = (
                _shuffle_fields(fields, seed ^ (i * 999_983))
                if self._shuffle else fields
            )
            cs = CutSet.from_shar(fields=effective, shuffle_shards=False)
            cs = cs.modify_ids(lambda cid, p=id_prefix: f"{p}-{cid}")
            cutsets.append(cs)
            weights.append(weight)

        if len(cutsets) == 1:
            yield from cutsets[0]
        else:
            yield from CutSet.mux(*cutsets, weights=weights, seed=abs(seed), stop_early=False)


# ---------------------------------------------------------------------------
# Cyclic auxiliary cut generator  (shard-level shuffle, one tar open at a time)
# ---------------------------------------------------------------------------

def _cyclic_shard_gen(
    sources:     List[ShardSource],
    field_names: List[str],
    base_seed:   int,
    shuffle:     bool = True,
):
    """
    Module-level infinite generator for one auxiliary source group.

    Discovers shards lazily (post-fork) on the first call.  On each pass,
    shuffles the shard list with a pass-unique seed, then reads cuts
    sequentially from each shard's tar files (one shard open at a time).
    No per-cut shuffle buffer is maintained.

    Kept module-level (not a bound method) to avoid reference cycles
    between _CyclicCutIterator and this generator that would prevent
    release_auxiliary() from freeing memory promptly.
    """
    # Lazy discovery — runs once, post-fork.
    discovered: List[Tuple[Dict[str, List[str]], str, float]] = []
    for src in sources:
        fields    = _discover_shards(src.shard_dir, field_names)
        id_prefix = Path(src.shard_dir).name
        discovered.append((fields, id_prefix, src.weight))

    pass_num = 0
    while True:
        seed = abs(base_seed + pass_num * 99_991)

        cutsets: List[CutSet] = []
        weights: List[float]  = []
        for i, (fields, id_prefix, weight) in enumerate(discovered):
            effective = (
                _shuffle_fields(fields, seed ^ (i * 999_983))
                if shuffle else fields
            )
            cs = CutSet.from_shar(fields=effective, shuffle_shards=False)
            cs = cs.modify_ids(lambda cid, p=id_prefix: f"{p}-{cid}")
            cutsets.append(cs)
            weights.append(weight)

        if os.getenv("SHARD_DEBUG"):
            worker_id, _ = _get_worker_info()
            if len(cutsets) == 1:
                label = f"aux:{discovered[0][1]}"
                yield from _timed_gen(cutsets[0], label=label, worker_id=worker_id)
            else:
                cs_mux = CutSet.mux(*cutsets, weights=weights, seed=seed, stop_early=False)
                yield from _timed_gen(cs_mux, label="aux:mux", worker_id=worker_id)
        else:
            if len(cutsets) == 1:
                yield from cutsets[0]
            else:
                yield from CutSet.mux(*cutsets, weights=weights, seed=seed, stop_early=False)

        pass_num += 1


class _CyclicCutIterator:
    """
    Infinite iterator over auxiliary shard cuts.

    Backed by _cyclic_shard_gen (shard-level shuffle, sequential tar reads).
    When aux_buffer_size > 0 a fixed-size reservoir holds pre-fetched cuts;
    next() picks a random slot and replaces it, giving better in-epoch
    randomisation than pure sequential order at the cost of aux_buffer_size
    cuts in RAM.

    Shard paths are discovered once inside _cyclic_shard_gen on the first
    iteration; subsequent passes reuse them without further glob calls.
    Must be instantiated post-fork (inside __iter__).
    """

    def __init__(
        self,
        sources:         List[ShardSource],
        field_names:     List[str],
        base_seed:       int,
        shuffle:         bool = True,
        aux_buffer_size: int  = 0,
    ):
        self._aux_buffer_size = aux_buffer_size
        self._iter            = _cyclic_shard_gen(sources, field_names, base_seed, shuffle)
        self._buffer: List[Cut] = [next(self._iter) for _ in range(aux_buffer_size)]

    def next(self) -> Cut:
        if self._aux_buffer_size > 0:
            k = random.randrange(self._aux_buffer_size)
            cut = self._buffer[k]
            self._buffer[k] = next(self._iter)
            return cut
        return next(self._iter)

    def next_n(self, n: int) -> List[Cut]:
        if self._aux_buffer_size > 0 and n <= self._aux_buffer_size:
            indices = random.sample(range(self._aux_buffer_size), n)
            cuts    = [self._buffer[k] for k in indices]
            for k in indices:
                self._buffer[k] = next(self._iter)
            return cuts
        return [self.next() for _ in range(n)]


# ---------------------------------------------------------------------------
# Audio loading / padding helpers
# ---------------------------------------------------------------------------

def _to_numpy(audio: np.ndarray) -> np.ndarray:
    """Ensure shape (C, T) with C ≥ 1."""
    return audio[np.newaxis, :] if audio.ndim == 1 else audio


def _pad_to(audio: np.ndarray, n: int) -> np.ndarray:
    """Right-zero-pad or truncate to exactly n samples."""
    audio = _to_numpy(audio)
    T = audio.shape[-1]
    if T >= n:
        return audio[:, :n]
    return np.pad(audio, ((0, 0), (0, n - T)))


def _sync_trim_or_pad(arrays: List[np.ndarray], n: int) -> List[np.ndarray]:
    """
    Trim or pad all arrays with the SAME random offset so paired signals
    (farend + echo) remain temporally aligned.
    """
    T = arrays[0].shape[-1]
    if T >= n:
        return [a[:, :n] for a in arrays]
    pad   = n - T
    left  = random.randint(0, pad)
    right = pad - left
    return [np.pad(a, ((0, 0), (left, right))) for a in arrays]


def _trim_or_pad_noise(audio: np.ndarray, n: int) -> np.ndarray:
    """Random-offset crop or right-zero-pad for noise signals."""
    audio = _to_numpy(audio)
    T = audio.shape[-1]
    if T >= n:
        start = random.randint(0, T - n)
        return audio[:, start:start + n]
    return np.pad(audio, ((0, 0), (0, n - T)))


def _wrap_crop(audio: np.ndarray, cursor: int, n: int) -> np.ndarray:
    """
    Return n samples from audio starting at cursor, wrapping around if
    the end of the array is reached.  Tiles audio if n > len(audio).
    """
    T = audio.shape[-1]
    if T == 0:
        return np.zeros((audio.shape[0], n), dtype=np.float32)
    cursor = cursor % T
    total_needed = cursor + n
    if total_needed <= T:
        return audio[:, cursor:cursor + n]
    reps = math.ceil(total_needed / T)
    tiled = np.tile(audio, (1, reps))
    return tiled[:, cursor:cursor + n]


def _numpy_stack(
    arrays:     List[Optional[np.ndarray]],
    target_len: int,
) -> Tensor:
    """
    Stack optional (C, T_i) arrays into a (B, C, target_len) tensor.
    None entries become zeros.  Always returns a tensor.
    """
    C_ref = next((a.shape[0] for a in arrays if a is not None), 1)
    stacked = [
        _pad_to(a, target_len) if a is not None
        else np.zeros((C_ref, target_len), dtype=np.float32)
        for a in arrays
    ]
    return torch.from_numpy(np.stack(stacked)).float()


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------

@dataclass
class ShardBatch:
    """
    Audio, metadata, and optional debug IDs from ShardDataset.

    Audio fields are always tensors when their corresponding prob > 0.
    Items for which the dataset decided not to load are zero-filled.

    speech       (B, 1, T)      codec or clean speech.
    speech_clean (B, 1, T)      always clean; present when prob_speech_codec > 0.
    farend       (B, 1, T)      far-end reference; zeros for non-echo items.
    echo         (B, 1, T)      acoustic echo aligned with farend; zeros otherwise.
    noise        (B, 1, T)      noise clip; zeros when not loaded.
    rir          (B, 1, T_rir)  RIR or unit impulse [1, 0, …].
    rir_onset    (B,) int64     onset sample (0 for unit impulse).
    dbFS         (B,) float32   target loudness in dBFS per item.
                                Populated by validation ShardDataset; None otherwise.
    snr          (B,) float32   target SNR for noise mixing (inf = no noise).
                                Populated by validation ShardDataset; None otherwise.
    ser          (B,) float32   target SER for echo mixing (inf = no echo).
                                Populated by validation ShardDataset; None otherwise.
    is_real      (B,) bool      True when a real echo pair was used.
    num_samples  (B,) int64     unpadded speech sample counts.
    text         List[str]      selected text variant (see text_field).
    id_speech    List[str]      cut.id of each speech cut (debug).
    id_noise     List[str]      noise cut.id or "" (debug).
    id_farend_echo List[str]    fe cut.id or "" (debug).
    id_rir       List[str]      rir cut.id or "" (debug).
    """
    speech:         Optional[torch.Tensor] = None
    speech_clean:   Optional[torch.Tensor] = None
    farend:         Optional[torch.Tensor] = None
    echo:           Optional[torch.Tensor] = None
    noise:          Optional[torch.Tensor] = None
    rir:            Optional[torch.Tensor] = None
    rir_onset:      Optional[torch.Tensor] = None
    rir_t60:        Optional[torch.Tensor] = None
    rms_speech:     Optional[torch.Tensor] = None
    rms_echo:       Optional[torch.Tensor] = None
    rms_noise:      Optional[torch.Tensor] = None
    dbFS:           Optional[torch.Tensor] = None
    snr:            Optional[torch.Tensor] = None
    ser:            Optional[torch.Tensor] = None
    is_real:        Optional[torch.Tensor] = None
    num_samples:    Optional[torch.Tensor] = None
    text:           Optional[List[str]]    = None
    id_speech:      Optional[List[str]]    = None
    id_noise:       Optional[List[str]]    = None
    id_farend_echo: Optional[List[str]]    = None
    id_rir:         Optional[List[str]]    = None

    def to(self, device: torch.device) -> "ShardBatch":
        """Move all tensors to device; strings, lists, and None pass through."""
        def _mv(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x
        return ShardBatch(**{f.name: _mv(getattr(self, f.name))
                             for f in dc_fields(self)})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardDataset(IterableDataset):
    """
    DDP + multi-worker safe Lhotse shard dataset for any audio/speech task.

    Epoch control and step counting
    ─────────────────────────────────
    The dataset yields batches infinitely.  The training loop controls epoch
    length by counting steps and breaking:

        for epoch in range(num_epochs):
            dataset.set_epoch(epoch)
            for step, batch in zip(range(steps_per_epoch), loader):
                ...

    set_epoch(e) updates a shared mp.Value so all persistent workers read
    the new seed at the start of the next epoch's __iter__ call.

    Lazy shard discovery
    ─────────────────────
    No filesystem glob calls happen in __init__.  Discovery runs once per
    worker (post-fork) on the first __iter__ call and is cached thereafter.

    Shard-level shuffling
    ──────────────────────
    At every epoch, each worker shuffles its local list of shard file paths
    using its unique seed.  CutSet.from_shar() then reads shards in that
    shuffled order, opening one shard's tar files at a time.  No per-cut
    shuffle reservoir is maintained.

    text_field
    ───────────
    Selects which text variant to return in ShardBatch.text:
      "text"   → cut.supervisions[0].text  (raw transcript)
      anything else → cut.supervisions[0].custom[text_field]
    """

    def __init__(
        self,
        speech_sources:      List[ShardSource],
        farend_echo_sources: Optional[List[ShardSource]] = None,
        noise_sources:       Optional[List[ShardSource]] = None,
        rir_sources:         Optional[List[ShardSource]] = None,
        max_duration:        float = 30.0,
        sampling_rate:       int   = 16_000,
        shuffle:             bool  = True,
        num_buckets:         int   = 20,
        cut_buffer_size:     int   = 1_000,   # kept for API compatibility; unused in v3
        aux_buffer_size:     int   = 0,
        sampler_buffer_size: int   = 20_000,
        base_seed:           int   = 42,
        max_utt_duration:    Optional[float] = None,
        min_utt_duration:    Optional[float] = None,
        ids_to_filter:       Optional[List[str]] = None,
        prob_speech_codec:   float = 0.0,
        prob_speech_reverb:  float = 0.0,
        prob_farend_echo:    float = 0.0,
        prob_farend_only:    float = 0.0,
        prob_noise:          float = 0.0,
        text_field:          str   = "text",
        pad_mode:            str   = "zeros",
        keys: tp.List[str]        = ("speech", "num_samples"),
    ):
        super().__init__()

        assert prob_speech_codec + prob_speech_reverb <= 1.0 + 1e-6, \
            "prob_speech_codec + prob_speech_reverb must be ≤ 1.0"
        invalid = set(keys) - VALID_KEYS
        assert not invalid, f"Invalid keys: {invalid}"

        self.speech_sources      = speech_sources
        self.farend_echo_sources = farend_echo_sources or []
        self.noise_sources       = noise_sources or []
        self.rir_sources         = rir_sources or []
        self.max_duration        = max_duration
        self.sampling_rate       = sampling_rate
        self._shuffle            = shuffle
        self.num_buckets         = num_buckets
        self.cut_buffer_size     = cut_buffer_size
        self.aux_buffer_size     = aux_buffer_size
        self.sampler_buffer_size = sampler_buffer_size
        self.max_utt_duration    = max_utt_duration
        self.min_utt_duration    = min_utt_duration
        self.ids_to_filter       = set(ids_to_filter) if ids_to_filter else set()
        self.text_field          = text_field

        self.prob_speech_codec  = prob_speech_codec
        self.prob_speech_reverb = prob_speech_reverb
        self.prob_farend_echo   = prob_farend_echo
        self.prob_farend_only   = prob_farend_only
        self.prob_noise         = prob_noise

        assert pad_mode in ("zeros", "repeat"), \
            f"pad_mode must be 'zeros' or 'repeat', got {pad_mode!r}"
        self.pad_mode = pad_mode

        # ── Seed management ───────────────────────────────────────────────────
        self._base_seed  = base_seed
        self._epoch_seed = mp.Value('i', base_seed)
        self._iter_count = 0

        # ── Auxiliary-release flag (shared across workers) ────────────────────
        self._release_auxiliary_flag = mp.Value('b', False)

        # ── Effective key set ─────────────────────────────────────────────────
        self.keys: tp.Set[str] = set(keys)
        if prob_speech_reverb > 0:                       self.keys.add("rir")
        if prob_farend_echo > 0 or prob_farend_only > 0: self.keys.update({"farend", "echo"})
        if prob_noise > 0:                               self.keys.add("noise")

        # ── Iterator / loader flags ───────────────────────────────────────────
        self._need_fe    = prob_farend_echo > 0 or prob_farend_only > 0
        self._need_noise = prob_noise > 0
        self._need_rir   = prob_speech_reverb > 0
        self._need_codec = prob_speech_codec > 0

        # ── Shard field lists ─────────────────────────────────────────────────
        self._speech_fields: List[str] = ["recording"] + \
                                         (["codec"] if self._need_codec else [])
        self._fe_fields: List[str] = (
            ["recording"] + (["echo"] if prob_farend_echo > 0 else [])
        ) if self._need_fe else []
        self._noise_fields: List[str] = ["recording"] if self._need_noise else []
        self._rir_fields:   List[str] = ["recording"] if self._need_rir   else []

        # Capture DDP rank NOW (pre-fork).
        self._ddp_rank = _get_rank()

        # ── Lazy-initialised per-worker state ─────────────────────────────────
        self._worker_key:     Optional[Tuple[int, int]]         = None
        self._worker_id:      int                               = 0
        self._speech_source:  Optional[_SpeechCutSource]        = None
        self._sampler:        Optional[DynamicBucketingSampler] = None
        self._fe_iter:        Optional[_CyclicCutIterator]      = None
        self._noise_iter:     Optional[_CyclicCutIterator]      = None
        self._rir_iter:       Optional[_CyclicCutIterator]      = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Update shared seed for the next epoch. Call from the main process."""
        with self._epoch_seed.get_lock():
            self._epoch_seed.value = self._base_seed + epoch

    def shuffle(self, seed: int) -> None:
        """Reseed workers for the next epoch. Alias for set_epoch(seed)."""
        with self._epoch_seed.get_lock():
            self._epoch_seed.value = seed

    def release_auxiliary(self) -> None:
        """
        Signal workers to drop their auxiliary iterators (fe/noise/rir) on the
        next __iter__ call.  Call from the main process before validation so the
        aux reservoir buffers are freed while workers are idle.  The sampler is
        NOT released (rebuilding it is expensive; set_epoch() reseeds it cheaply).
        """
        with self._release_auxiliary_flag.get_lock():
            self._release_auxiliary_flag.value = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_sampler(self, worker_seed: int) -> DynamicBucketingSampler:
        """
        Build the speech sampler.  Called at most once per worker lifetime;
        subsequent reshuffles use sampler.set_epoch() + _speech_source.set_epoch()
        which are cheap (no re-scan, only a new shard permutation).
        """
        self._speech_source = _SpeechCutSource(
            sources=self.speech_sources,
            field_names=self._speech_fields,
            shuffle=self._shuffle,
        )
        self._speech_source.set_epoch(worker_seed)
        speech_cuts = CutSet(cuts=self._speech_source)

        if self.max_utt_duration is not None:
            max_d = self.max_utt_duration
            def filter_max(c): return c.duration <= max_d
            speech_cuts = speech_cuts.filter(filter_max)
        if self.min_utt_duration is not None:
            min_d = self.min_utt_duration
            def filter_min(c): return c.duration >= min_d
            speech_cuts = speech_cuts.filter(filter_min)
        if self.ids_to_filter:
            ids = self.ids_to_filter
            def filter_ids(c): return c.id not in ids
            speech_cuts = speech_cuts.filter(filter_ids)

        if os.getenv("SHARD_DEBUG"):
            worker_id, _ = _get_worker_info()
            speech_cuts = CutSet(cuts=_TimedCuts(
                speech_cuts, label="speech", worker_id=worker_id,
            ))

        return DynamicBucketingSampler(
            speech_cuts,
            max_duration=self.max_duration,
            num_buckets=self.num_buckets,
            buffer_size=self.sampler_buffer_size,
            shuffle=self._shuffle,
            seed=worker_seed,
            rank=0,
            world_size=1,
        )

    def _build_cyclic(
        self,
        sources: List[ShardSource],
        fields:  List[str],
        seed:    int,
        offset:  int,
    ) -> _CyclicCutIterator:
        return _CyclicCutIterator(
            sources=sources,
            field_names=fields,
            base_seed=seed + offset,
            shuffle=self._shuffle,
            aux_buffer_size=self.aux_buffer_size,
        )

    def _get_text(self, cut: Cut) -> str:
        if not cut.supervisions:
            return ""
        sup    = cut.supervisions[0]
        custom = sup.custom or {}
        if self.text_field == "text":
            return sup.text or ""
        key  = "ipa" if self.text_field == "ipa_filtered" else self.text_field
        text = custom.get(key, "")
        if self.text_field == "ipa_filtered":
            text = filter_ipa(text)
        return text

    # ── Main iteration ────────────────────────────────────────────────────────

    def __iter__(self) -> Generator[ShardBatch, None, None]:
        rank       = self._ddp_rank
        worker_id, num_workers = _get_worker_info()

        # Release auxiliary iterators if flagged (set by main process before
        # validation).  Frees reservoir-buffer memory while workers are idle.
        with self._release_auxiliary_flag.get_lock():
            if self._release_auxiliary_flag.value:
                self._fe_iter    = None
                self._noise_iter = None
                self._rir_iter   = None
                self._release_auxiliary_flag.value = False

        with self._epoch_seed.get_lock():
            init_seed = self._epoch_seed.value

        self._iter_count += 1
        worker_seed = init_seed + self._iter_count + worker_id + 1_000 * rank

        random.seed(worker_seed)
        np.random.seed(worker_seed)

        worker_key    = (rank, worker_id)
        is_new_worker = self._worker_key != worker_key

        # ── Lazy init: speech sampler (expensive first build, cheap thereafter)
        if self._sampler is None or is_new_worker:
            self._sampler = self._build_sampler(worker_seed)
        else:
            # Cheap epoch reshuffle: update shard permutation seed, then reseed
            # the sampler's internal RNG without rebuilding the bucket histogram.
            self._speech_source.set_epoch(worker_seed)
            self._sampler.set_epoch(worker_seed)

        # ── Lazy init: auxiliary iterators ────────────────────────────────────
        if self._need_fe and self.farend_echo_sources:
            if self._fe_iter is None or is_new_worker:
                self._fe_iter = self._build_cyclic(
                    self.farend_echo_sources, self._fe_fields,
                    seed=worker_seed, offset=111_111,
                )
        if self._need_noise and self.noise_sources:
            if self._noise_iter is None or is_new_worker:
                self._noise_iter = self._build_cyclic(
                    self.noise_sources, self._noise_fields,
                    seed=worker_seed, offset=222_222,
                )
        if self._need_rir and self.rir_sources:
            if self._rir_iter is None or is_new_worker:
                self._rir_iter = self._build_cyclic(
                    self.rir_sources, self._rir_fields,
                    seed=worker_seed, offset=333_333,
                )

        self._worker_key = worker_key
        self._worker_id  = worker_id

        _debug = bool(os.getenv("SHARD_DEBUG"))
        pass_num = 0
        while True:
            _t_sampler = time.perf_counter()
            for speech_batch in self._sampler:
                sampler_time = time.perf_counter() - _t_sampler
                yield from self._process_batch(speech_batch, sampler_time, _debug)
                _t_sampler = time.perf_counter()

            pass_num += 1
            print(f"all speech exhausted")
            self._speech_source.set_epoch(worker_seed + pass_num * 99_991)
            self._sampler.set_epoch(worker_seed + pass_num * 99_991)

    def _process_batch(
        self, speech_batch, sampler_time: float = 0.0, debug: bool = False,
    ) -> Generator[ShardBatch, None, None]:
        """Convert one sampler mini-batch into a ShardBatch and yield it."""
        cuts: List[Cut] = list(speech_batch)
        B               = len(cuts)
        max_dur         = max(c.duration for c in cuts)
        target_samples  = int(np.ceil(max_dur * self.sampling_rate))

        # ── Step 1: per-item decisions (no I/O) ───────────────────────────────
        use_codec  = [False] * B
        use_reverb = [False] * B
        echo_type  = ["none"] * B   # "full" | "farend_only" | "none"

        for i in range(B):
            r = random.random()
            if r < self.prob_speech_codec:
                use_codec[i] = True
            elif r < self.prob_speech_codec + self.prob_speech_reverb:
                use_reverb[i] = True

            r = random.random()
            if r < self.prob_farend_echo:
                echo_type[i] = "full"
            elif r < self.prob_farend_echo + self.prob_farend_only:
                echo_type[i] = "farend_only"

        # ── Step 2: draw fe pool ──────────────────────────────────────────────
        _t0 = time.perf_counter()
        fe_needed  = [i for i in range(B) if echo_type[i] != "none"]
        fe_pool:   List[Cut] = []
        if fe_needed and self._fe_iter:
            fe_dur_needed = len(fe_needed) * max_dur
            pool_dur = 0.0
            while pool_dur < fe_dur_needed and len(fe_pool) < len(fe_needed):
                fc = self._fe_iter.next()
                fe_pool.append(fc)
                pool_dur += fc.duration
        _t_fe_draw = time.perf_counter() - _t0

        # ── Step 3: is_real from pool (round-robin; no audio I/O) ────────────
        is_real_list = [False] * B
        if fe_pool:
            for j, i in enumerate(fe_needed):
                is_real_list[i] = bool(
                    fe_pool[j % len(fe_pool)].custom.get("is_real", False))

        # ── Step 4: noise decisions (is_real now known) ───────────────────────
        noise_needed: List[int] = []
        for i in range(B):
            if not (is_real_list[i] and echo_type[i] == "full"):
                if random.random() < self.prob_noise:
                    noise_needed.append(i)

        # ── Step 5: draw noise pool and RIR cuts ─────────────────────────────
        _t0 = time.perf_counter()
        noise_pool: List[Cut] = []
        if noise_needed and self._noise_iter:
            noise_dur_needed = len(noise_needed) * max_dur
            pool_dur = 0.0
            while pool_dur < noise_dur_needed and len(noise_pool) < len(noise_needed):
                nc = self._noise_iter.next()
                noise_pool.append(nc)
                pool_dur += nc.duration

        rir_needed = [i for i in range(B) if use_reverb[i]]
        rir_drawn  = self._rir_iter.next_n(len(rir_needed)) \
                     if self._rir_iter and rir_needed else []
        rir_cuts: List[Optional[Cut]] = [None] * B
        for slot, idx in enumerate(rir_needed):
            rir_cuts[idx] = rir_drawn[slot]
        rir_onset_list = [int(c.custom["onset_sample"])
                          if c else 0 for c in rir_cuts]
        rir_t60_list   = [float(c.custom["t60"])
                          if c else -1.0 for c in rir_cuts]
        _t_noise_draw = time.perf_counter() - _t0

        _print_timing = debug and sampler_time > 1.0

        # ── Step 6: load speech ───────────────────────────────────────────────
        _t0 = time.perf_counter()
        speech_arrays:       List[np.ndarray] = []
        speech_clean_arrays: List[np.ndarray] = []

        for i, cut in enumerate(cuts):
            if use_codec[i]:
                speech_arrays.append(_to_numpy(cut.codec.load_audio()))
                speech_clean_arrays.append(_to_numpy(cut.load_audio()))
            else:
                arr = _to_numpy(cut.load_audio())
                speech_arrays.append(arr)
                speech_clean_arrays.append(arr)

        num_samples_list: List[int] = [a.shape[-1] for a in speech_clean_arrays]
        _t_speech = time.perf_counter() - _t0

        # ── Step 7: load farend + echo from pooled cuts ───────────────────────
        _t0 = time.perf_counter()
        farend_arrays: List[Optional[np.ndarray]] = [None] * B
        echo_arrays:   List[Optional[np.ndarray]] = [None] * B

        if fe_pool:
            need_echo     = any(echo_type[i] == "full" for i in fe_needed)
            farend_stream = np.concatenate(
                [_to_numpy(fc.load_audio()) for fc in fe_pool], axis=-1)
            echo_stream   = np.concatenate(
                [_to_numpy(fc.echo.load_audio()) for fc in fe_pool], axis=-1
            ) if need_echo else None

            T_fe      = farend_stream.shape[-1]
            fe_cursor = random.randint(0, max(0, T_fe - target_samples))
            for i in fe_needed:
                farend_arrays[i] = _wrap_crop(farend_stream, fe_cursor, target_samples)
                if echo_type[i] == "full" and echo_stream is not None:
                    echo_arrays[i] = _wrap_crop(echo_stream, fe_cursor, target_samples)
                fe_cursor = (fe_cursor + target_samples) % T_fe if T_fe > 0 else 0
        _t_farend = time.perf_counter() - _t0

        # ── Step 8: load noise from pooled cuts ───────────────────────────────
        _t0 = time.perf_counter()
        noise_arrays: List[Optional[np.ndarray]] = [None] * B

        if noise_pool:
            noise_stream = np.concatenate(
                [_to_numpy(nc.load_audio()) for nc in noise_pool], axis=-1)
            T_noise      = noise_stream.shape[-1]
            noise_cursor = random.randint(0, max(0, T_noise - 1))
            for i in noise_needed:
                noise_arrays[i] = _wrap_crop(noise_stream, noise_cursor, target_samples)
                noise_cursor = (noise_cursor + target_samples) % T_noise \
                               if T_noise > 0 else 0
        _t_noise = time.perf_counter() - _t0

        # ── Step 9: load RIR (unit impulse for non-reverb items) ─────────────
        _t0 = time.perf_counter()
        rir_arrays: List[np.ndarray] = [
            _to_numpy(rir_cuts[i].load_audio()) if rir_cuts[i] is not None
            else np.array([[1.0]], dtype=np.float32)
            for i in range(B)
        ]
        _t_rir = time.perf_counter() - _t0

        if _print_timing:
            print(
                f"  [W{self._worker_id}] sampler={sampler_time:.3f}s"
                f"  fe_draw={_t_fe_draw:.3f}s  noise_draw={_t_noise_draw:.3f}s"
                f"  speech={_t_speech:.3f}s  farend={_t_farend:.3f}s"
                f"  noise={_t_noise:.3f}s  rir={_t_rir:.3f}s  B={B}"
                f"  fe_pool={len(fe_pool)}/{len(fe_needed)}"
                f"  noise_pool={len(noise_pool)}/{len(noise_needed)}",
                flush=True,
            )

        # ── Step 10: stack into tensors ───────────────────────────────────────
        if self.pad_mode == "repeat":
            def _stack(arr_list: List[np.ndarray]) -> Tensor:
                return torch.from_numpy(
                    np.stack([_wrap_crop(a, 0, target_samples) for a in arr_list])
                ).float()
        else:
            def _stack(arr_list: List[np.ndarray]) -> Tensor:
                return torch.from_numpy(
                    np.stack([_pad_to(a, target_samples) for a in arr_list])
                ).float()

        if any(use_codec):
            speech_t       = _stack(speech_arrays);       del speech_arrays
            speech_clean_t = _stack(speech_clean_arrays); del speech_clean_arrays
        else:
            speech_t = _stack(speech_arrays)
            del speech_arrays, speech_clean_arrays
            speech_clean_t = speech_t
        farend_t = _numpy_stack(farend_arrays, target_samples); del farend_arrays
        echo_t   = _numpy_stack(echo_arrays,   target_samples); del echo_arrays
        noise_t  = _numpy_stack(noise_arrays,  target_samples); del noise_arrays

        max_rir = max(a.shape[-1] for a in rir_arrays)
        rir_t   = _numpy_stack(rir_arrays, max_rir); del rir_arrays

        # ── Step 11: text and debug IDs ───────────────────────────────────────
        texts: Optional[List[str]] = None
        if "text" in self.keys:
            texts = [self._get_text(c) for c in cuts]

        id_speech = [c.id for c in cuts] \
                    if "id_speech" in self.keys else None

        if "id_noise" in self.keys:
            _noise_cut_per_item: List[Optional[Cut]] = [None] * B
            for j, i in enumerate(noise_needed):
                if noise_pool:
                    _noise_cut_per_item[i] = noise_pool[j % len(noise_pool)]
            id_noise: Optional[List[str]] = [
                c.id if c else "" for c in _noise_cut_per_item
            ]
        else:
            id_noise = None

        if "id_farend_echo" in self.keys:
            _fe_cut_per_item: List[Optional[Cut]] = [None] * B
            for j, i in enumerate(fe_needed):
                if fe_pool:
                    _fe_cut_per_item[i] = fe_pool[j % len(fe_pool)]
            id_fe: Optional[List[str]] = [
                c.id if c else "" for c in _fe_cut_per_item
            ]
        else:
            id_fe = None

        id_rir = [rir_cuts[i].id if rir_cuts[i] else ""
                  for i in range(B)] \
                 if "id_rir" in self.keys else None

        del fe_pool, noise_pool

        yield ShardBatch(
            speech          = speech_t,
            speech_clean    = speech_clean_t if "speech_clean"   in self.keys else None,
            farend          = farend_t       if "farend"         in self.keys else None,
            echo            = echo_t         if "echo"           in self.keys else None,
            noise           = noise_t        if "noise"          in self.keys else None,
            rir             = rir_t          if "rir"            in self.keys else None,
            rir_onset       = torch.tensor(rir_onset_list,   dtype=torch.long)
                              if "rir_onset"   in self.keys else None,
            rir_t60         = torch.tensor(rir_t60_list,     dtype=torch.float32)
                              if "rir_t60"     in self.keys else None,
            is_real         = torch.tensor(is_real_list,     dtype=torch.bool)
                              if self._need_fe else None,
            num_samples     = torch.tensor(num_samples_list, dtype=torch.long)
                              if "num_samples" in self.keys else None,
            text            = texts,
            id_speech       = id_speech,
            id_noise        = id_noise,
            id_farend_echo  = id_fe,
            id_rir          = id_rir,
        )


# ---------------------------------------------------------------------------
# Batched grouped RIR convolution
# ---------------------------------------------------------------------------

def _batch_convolve_rir(speech: Tensor, rirs: Tensor) -> Tensor:
    """
    Causal linear convolution of speech[b] with rir[b] for all b at once.

    speech : (B, 1, T)
    rirs   : (B, 1, T_rir)   not yet flipped
    returns: (B, 1, T)

    Unit impulse rir = [1, 0, …, 0] → identity: y[n] = speech[n].
    """
    T     = speech.shape[-1]
    T_rir = rirs.shape[-1]

    if T_rir == 1:
        return speech * rirs

    target = T + T_rir - 1
    n = 1 << (target - 1).bit_length()

    Y = torch.fft.rfft(speech, n=n) * torch.fft.rfft(rirs, n=n)
    return torch.fft.irfft(Y, n=n)[..., :T]


# ---------------------------------------------------------------------------
# RIR target generators (for dereverberation)
# ---------------------------------------------------------------------------

def get_weighted_rir(
    rirs:         Tensor,
    onset_sample: Tensor,
    t60_max:      float = 0.3,
    fs:           int = 16000,
) -> Tensor:
    """Apply exponential decay to RIR based on desired reverberation time.

    w(t) = 1                                    for t <= t0
           exp(-(t - t0) * 6*log(10) / t60_max) for t >  t0

    Args:
        rirs:         [B, 1, T]  batch of (zero-padded) RIRs
        onset_sample: [B]        onset sample index per RIR
        t60_max:      desired reverberation time in seconds
        fs:           sampling frequency in Hz
    Returns:
        weighted_rirs: [B, 1, T]
    """
    B, _, T = rirs.shape
    device = rirs.device

    t  = torch.arange(T, dtype=torch.float32, device=device).view(1, 1, T) / fs
    t0 = onset_sample.float().view(B, 1, 1) / fs

    decay  = torch.exp(-(t - t0) * 6.0 * math.log(10) / t60_max)
    weight = torch.where(t > t0, decay, torch.ones_like(decay))

    return rirs * weight


def get_rts_rir(
    rirs:     Tensor,
    onset:    Tensor,
    t60:      Tensor,
    t60_max:  float = 0.15,
    fs:       int = 16000,
) -> Tensor:
    """Apply Reverberation Time Shortening (https://arxiv.org/abs/2204.08765).

    w(t) = 1                                    for t <= t0
           10**(-3/fs*(1/t60_max - 1/t60) * (n-n0)) for t >  t0

    Args:
        rirs:     [B, 1, T]  batch of (zero-padded) RIRs
        onset:    [B]        onset sample index per RIR
        t60:      [B]        desired reverberation time in seconds
        t60_max:  desired reverberation time in seconds
        fs:       sampling frequency in Hz
    Returns:
        weighted_rirs: [B, 1, T]
    """
    B, _, T = rirs.shape
    device = rirs.device
    onset = onset.view(B, 1, 1)
    t60   = t60.view(B, 1, 1)

    n = torch.arange(T, dtype=torch.float32, device=device).view(1, 1, T)

    q     = 3 / fs * (1 / t60_max - 1 / t60)
    decay = 10 ** (-q * (n - onset))
    weight = torch.where(n > onset, decay, torch.ones_like(decay))
    return torch.where(t60 <= t60_max, rirs, rirs * weight)


def get_early_rir(
    rirs: Tensor,
    onset_sample: Tensor,
    early_rir_sec: float = 0.05,
    fs: int = 16000,
) -> Tensor:
    """Zero out everything beyond the early reflection window.

    Keeps samples in [0, onset_sample + early_rir_samples) and zeros the rest.

    Args:
        rirs:          [B, 1, T]  batch of (zero-padded) RIRs
        onset_sample:  [B]        onset sample index per RIR
        early_rir_sec: duration in seconds counted as early RIR
        fs:            sampling frequency in Hz
    Returns:
        early_rirs: [B, 1, T]
    """
    B, _, T = rirs.shape
    device = rirs.device

    early_rir_samples = int(early_rir_sec * fs)
    stop_sample = (onset_sample + early_rir_samples).view(B, 1, 1)
    idx  = torch.arange(T, device=device).view(1, 1, T)
    mask = idx < stop_sample

    return rirs * mask.float()


# ---------------------------------------------------------------------------
# Mixer output
# ---------------------------------------------------------------------------

@dataclass
class MixedBatch:
    """
    Output of DynamicMixer.

    speech       (B, 1, T)  mixed noisy signal  — model input.
    speech_clean (B, 1, T)  reverbed + normalised clean signal  — model target.
                            None if batch.speech_clean was None.
    farend       (B, 1, T)  far-end reference; zeros for non-echo items.
                            None if batch.farend was None.
    num_samples  (B,)       unpadded speech sample counts.
                            None if batch.num_samples was None.
    text         List[str]  text strings; None if not in dataset keys.
    dbFS         (B,)       target loudness in dBFS; None if batch.dbFS was None.
    """
    speech:       Tensor               = None
    speech_clean: Optional[Tensor]     = None
    farend:       Optional[Tensor]     = None
    num_samples:  Optional[Tensor]     = None
    text:         Optional[List[str]]  = None
    dbFS:         Optional[Tensor]     = None
    snr:          Optional[Tensor]     = None
    ser:          Optional[Tensor]     = None


# ---------------------------------------------------------------------------
# DynamicMixer  — pure math, no probability parameters
# ---------------------------------------------------------------------------

class DynamicMixer:
    """
    GPU mixing engine.  All load/skip decisions live in ShardDataset.

    Pipeline
    ────────
    1. RIR convolution
        inp_speech = batch.speech * batch.rir  (full RIR → reverberant input)
        Unit-impulse RIR items pass through unchanged.

    2. speech_clean target  (only when batch.speech_clean is not None)
        rir_target_type = "early_rir"    → get_early_rir(batch.rir, batch.rir_onset)
        rir_target_type = "weighted_rir" → get_weighted_rir(batch.rir, batch.rir_onset)
        rir_target_type = "anechoic"     → unit impulse (anechoic clean speech)
        rir_target_type = None           → batch.rir (no dereverberation)
        speech_clean_out = batch.speech_clean * rir_target

    3. dBFS normalisation — RMS computed from inp_speech (reverberant),
        same scale applied to both inp_speech and speech_clean_out.

    4. Echo mixing   — zero echo tensor adds nothing.
    5. Noise mixing  — zero noise tensor adds nothing.
    6. Peak clip     — same denominator applied to speech_clean_out.

    Parameters
    ──────────
    rir_target_type     "early_rir" | "weighted_rir" | "anechoic" | "rts" | None
                        Controls the dereverberation target.  None = anechoic.
    early_rir_sec       Duration of the early RIR window in seconds (early_rir).
    t60_max             Desired T60 in seconds (weighted_rir).
    """

    def __init__(
        self,
        speech_dbFS:         tp.List[float]  = (-30.0, -15.0),
        ser_real:            tp.List[float]  = (-10.0,  20.0),
        ser:                 tp.List[float]  = (-10.0,  20.0),
        snr:                 tp.List[float]  = (  0.0,  30.0),
        sampling_rate:       int             = 16_000,
        seg_window_ms:       int             = 100,
        seg_rel_threshold:   float           = -25.0,
        seg_abs_threshold:   Optional[float] = -50.0,
        rir_target_type:     Optional[str]   = None,
        early_rir_sec:       float           = 0.05,
        t60_max:             float           = 0.3,
    ):
        assert rir_target_type in (None, "early_rir", "weighted_rir", "anechoic", "rts"), \
            "rir_target_type must be None, 'early_rir', 'weighted_rir', 'anechoic', or 'rts'"
        self.dbFS            = list(speech_dbFS)
        self.ser_real        = list(ser_real)
        self.ser             = list(ser)
        self.snr             = list(snr)
        self.sr              = sampling_rate
        self.seg_window_ms   = seg_window_ms
        self.seg_rel_thr     = seg_rel_threshold
        self.seg_abs_thr     = seg_abs_threshold
        self.rir_target_type = rir_target_type
        self.early_rir_sec   = early_rir_sec
        self.t60_max         = t60_max

    @torch.no_grad()
    def __call__(self, batch: ShardBatch) -> MixedBatch:
        assert batch.speech is not None, "batch.speech must not be None"
        device  = batch.speech.device
        B, _, T = batch.speech.shape

        # ── 1. RIR convolution ─────────────────────────────────────────────────
        inp_speech = batch.speech
        if batch.rir is not None:
            inp_speech = _batch_convolve_rir(inp_speech, batch.rir)

        # ── 2. speech_clean target ─────────────────────────────────────────────
        speech_clean: Optional[Tensor] = None
        if batch.speech_clean is not None:
            if self.rir_target_type == "anechoic" or batch.rir is None:
                rir_target = None
            elif self.rir_target_type is None:
                rir_target = batch.rir
            elif self.rir_target_type == "early_rir":
                assert batch.rir_onset is not None, \
                    "rir_onset required for early_rir target"
                rir_target = get_early_rir(
                    batch.rir, batch.rir_onset,
                    early_rir_sec=self.early_rir_sec, fs=self.sr,
                )
            elif self.rir_target_type == "rts":
                assert batch.rir_onset is not None and batch.rir_t60 is not None, \
                    "rir_onset and rir_t60 required for rts target"
                rir_target = get_rts_rir(
                    batch.rir, batch.rir_onset, batch.rir_t60,
                    t60_max=self.t60_max, fs=self.sr,
                )
            else:  # "weighted_rir"
                assert batch.rir_onset is not None, \
                    "rir_onset required for weighted_rir target"
                rir_target = get_weighted_rir(
                    batch.rir, batch.rir_onset,
                    t60_max=self.t60_max, fs=self.sr,
                )

            if rir_target is not None:
                speech_clean = _batch_convolve_rir(batch.speech_clean, rir_target)

        # ── 3. dBFS normalisation ──────────────────────────────────────────────
        rms = segmental_rms(
            inp_speech, sr=self.sr,
            window_ms=self.seg_window_ms,
            relative_threshold_db=self.seg_rel_thr,
            absolute_threshold_db=None,
        )
        if not torch.isfinite(rms).all():
            idx = torch.where(~torch.isfinite(rms.view(-1)))[0]
            if batch.id_speech is not None:
                id = [batch.id_speech[i] for i in idx.tolist()]
                print(f"### Warning: non-finite RMS in batch with speech IDs: {id} ###")
            else:
                print("### Warning: non-finite RMS in batch with unknown speech IDs ###")

        dbFS_vals  = torch.empty(B, device=device).uniform_(*self.dbFS)
        target_rms = (10.0 ** (dbFS_vals / 20.0)).view(B, 1, 1)

        norm_scale = target_rms / rms.clamp(min=1e-10)
        inp_speech = inp_speech * norm_scale
        if speech_clean is not None:
            speech_clean = speech_clean * norm_scale

        # ── 4. Echo mixing ─────────────────────────────────────────────────────
        ser = None
        if batch.echo is not None:
            is_real  = batch.is_real.to(device) if batch.is_real is not None \
                       else torch.zeros(B, dtype=torch.bool, device=device)
            rms_echo = segmental_rms(batch.echo, sr=self.sr,
                                     window_ms=self.seg_window_ms,
                                     relative_threshold_db=self.seg_rel_thr,
                                     absolute_threshold_db=self.seg_abs_thr)

            ser_r = torch.empty(B, device=device).uniform_(*self.ser_real)
            ser_s = torch.empty(B, device=device).uniform_(*self.ser)
            ser   = torch.where(is_real, ser_r, ser_s)
            ser_v = ser.view(B, 1, 1)

            echo_scale = target_rms / (rms_echo * 10.0 ** (ser_v / 20.0)).clamp(min=1e-10)
            inp_speech = inp_speech + echo_scale * batch.echo

        # ── 5. Noise mixing ────────────────────────────────────────────────────
        snr = None
        if batch.noise is not None:
            rms_noise   = segmental_rms(batch.noise, sr=self.sr,
                                        window_ms=self.seg_window_ms,
                                        relative_threshold_db=self.seg_rel_thr,
                                        absolute_threshold_db=self.seg_abs_thr)
            snr         = torch.empty(B, device=device).uniform_(*self.snr)
            snr_v       = snr.view(B, 1, 1)
            noise_scale = target_rms / (rms_noise * 10.0 ** (snr_v / 20.0)).clamp(min=1e-10)
            inp_speech  = inp_speech + noise_scale * batch.noise

        # ── 6. Peak clip ───────────────────────────────────────────────────────
        max_abs = inp_speech.abs().amax(dim=(-2, -1), keepdim=True)
        if speech_clean is not None:
            max_abs = torch.maximum(max_abs, speech_clean.abs().amax(dim=(-2, -1), keepdim=True))
        clip_denom = torch.where(max_abs > 1.0, max_abs + 1e-5, torch.ones_like(max_abs))
        inp_speech = inp_speech / clip_denom
        if speech_clean is not None:
            speech_clean = speech_clean / clip_denom

        return MixedBatch(
            speech       = inp_speech,
            speech_clean = speech_clean,
            farend       = batch.farend,
            num_samples  = batch.num_samples,
            text         = batch.text,
            dbFS         = dbFS_vals,
            snr          = snr,
            ser          = ser,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_shard_dataloader(
    hparams: HParams,
    mode: str,
    keys: tp.List[str] = ("speech", "num_samples"),
) -> Tuple[ShardDataset, DataLoader]:
    """
    Build and return (dataset, dataloader).
    Call dataset.set_epoch(epoch) at the start of every training epoch.
    """
    hp: HParams = hparams.data[mode]

    prob_speech_reverb: float = hp.get("prob_speech_reverb", 0.0)
    prob_farend_echo:   float = hp.get("prob_farend_echo", 0.0)
    prob_farend_only:   float = hp.get("prob_farend_only", 0.0)
    prob_noise:         float = hp.get("prob_noise", 0.0)
    farend_echo_sources: Optional[List[ShardSource]] = None
    noise_sources:       Optional[List[ShardSource]] = None
    rir_sources:         Optional[List[ShardSource]] = None
    if (prob_farend_echo > 0 or prob_farend_only) > 0:
        farend_echo_sources = [ShardSource(**kwargs) for kwargs in hp.inputs.farend_echo]
    if prob_noise > 0:
        noise_sources       = [ShardSource(**kwargs) for kwargs in hp.inputs.noise]
    if prob_speech_reverb > 0:
        rir_sources         = [ShardSource(**kwargs) for kwargs in hp.inputs.rir]
    dataset = ShardDataset(
        speech_sources=[ShardSource(**kwargs) for kwargs in hp.inputs.speech],
        farend_echo_sources=farend_echo_sources,
        noise_sources=noise_sources,
        rir_sources=rir_sources,
        max_duration=hparams.data.max_duration,
        sampling_rate=hparams.data.sampling_rate,
        shuffle=hp.get("shuffle", True),
        num_buckets=hp.get("num_buckets", 20),
        cut_buffer_size=hp.get("cut_buffer_size", 1_000),
        aux_buffer_size=hp.get("aux_buffer_size", 0),
        sampler_buffer_size=hp.get("sampler_buffer_size", 20_000),
        base_seed=hparams.train.seed,
        max_utt_duration=hp.get("max_utt_duration", None),
        min_utt_duration=hp.get("min_utt_duration", None),
        ids_to_filter=hp.get("ids_to_filter", []),
        prob_speech_codec=hp.get("prob_speech_codec", 0.0),
        prob_speech_reverb=prob_speech_reverb,
        prob_farend_echo=prob_farend_echo,
        prob_farend_only=prob_farend_only,
        prob_noise=prob_noise,
        text_field=hparams.data.get("text", ""),
        pad_mode=hp.get("pad_mode", "zeros"),
        keys=keys,
    )
    def get(hp: HParams, attr: str, hp_fallback: HParams, default: Any) -> Any:
        return hp.get(attr, hp_fallback.get(attr, default))
    num_workers:        int = get(hp, "num_workers", hparams.data.train, 0)
    pin_memory:         int = get(hp, "pin_memory", hparams.data.train, False)
    persistent_workers: int = get(hp, "persistent_workers", hparams.data.train, False)
    prefetch_factor:    int = get(hp, "prefetch_factor", hparams.data.train, 2)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory if num_workers > 0 else False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import get_hparams
    hps = get_hparams("configs/asr_conv/korean/test.yaml")
    hps.data.max_duration = 100.0
    hps.data.train.num_buckets = 2
    hps.data.train.max_utt_duration = None
    hps.data.train.min_utt_duration = None
    dataset, loader = build_shard_dataloader(
        hparams = hps,
        mode    = "train",
        keys    = ["speech", "farend", "echo", "noise", "rir", "text", "speech_clean",
                   "rir_onset", "num_samples",
                   "id_speech", "id_noise", "id_farend_echo", "id_rir"],
    )

    mixer = DynamicMixer(**hps.data.train.mixer)

    for epoch in range(hps.train.max_epochs):
        dataset.set_epoch(epoch)
        for iteration, batch in enumerate(loader):
            batch = batch.to("cuda")
            mixed = mixer(batch)

            inp    = mixed.speech
            target = mixed.speech_clean
            fe     = mixed.farend
            lens   = mixed.num_samples

            breakpoint()
            if iteration == 2:
                break
