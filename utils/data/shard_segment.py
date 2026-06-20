"""
Shard-based fixed-segment dataset with on-the-fly loading.

ShardSegmentDataset — map-style Dataset that yields fixed-length audio
segments loaded from Lhotse shard format.  Returns ShardBatch (same class
as shard_on_the_fly_v3.ShardDataset) so it plugs into DynamicMixer unchanged.

Audio loading
─────────────
Mirrors NSOnTheFlyDataset.gen_audio(): pulls cuts from a cyclic iterator,
concatenating them with short silence gaps until segment_size samples are
accumulated, then random-crops the tail.  Farend+echo are loaded from the
same cut at the same crop offset to stay temporally aligned.

Infrastructure
──────────────
Follows shard_on_the_fly_v3 in every non-audio aspect:
  • set_epoch(epoch)       — updates mp.Value; propagates to persistent workers.
  • release_auxiliary()    — generation-counter mp.Value; each worker releases
                             its noise/fe/rir iterators on the next __getitem__.
  • Per-worker seeds       — epoch_seed + worker_id + 1000 * ddp_rank.
  • random / np.random     — reseeded at each epoch boundary per worker.
  • _build_cyclic helper   — mirrors ShardDataset._build_cyclic.
  • ShardBatch return type — per-item from __getitem__; _segment_collate_fn
                             stacks into a batched ShardBatch, padding RIR to
                             the batch maximum length.

DDP note
─────────
__getitem__ ignores idx; data order is driven by cyclic iterators seeded
per-worker, so DistributedSampler is not needed.  Each rank's workers have
different seeds, preventing duplicate data across GPUs.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import typing as tp
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils import data

from utils.data.shard_on_the_fly import (
    ShardBatch,
    ShardSource,
    _CyclicCutIterator,
    _get_rank,
    _to_numpy,
)


# ---------------------------------------------------------------------------
# Valid keys
# ---------------------------------------------------------------------------

VALID_KEYS: tp.FrozenSet[str] = frozenset({
    "speech",
    "speech_clean",
    "noise",
    "farend",
    "echo",
    "rir",
    "rir_onset",
    "rir_t60",
    "is_real",
    "id_speech",
    "id_noise",
    "id_farend_echo",
    "id_rir",
})


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _segment_collate_fn(samples: List[ShardBatch]) -> ShardBatch:
    """
    Collate per-item ShardBatches (shapes without a batch dim) into one
    batched ShardBatch.

      Audio (1, T)       →  (B, 1, T)         via torch.stack
      RIR   (1, T_rir)   →  (B, 1, max_T_rir) padded to batch maximum
      Scalar 0-d tensor  →  (B,)              via torch.stack
      str                →  List[str]
    """
    def _stack(key: str) -> Optional[Tensor]:
        vals = [getattr(s, key) for s in samples]
        if all(v is None for v in vals):
            return None
        return torch.stack([v for v in vals if v is not None])

    def _str_list(key: str) -> Optional[List[str]]:
        vals = [getattr(s, key) for s in samples]
        if all(v is None for v in vals):
            return None
        return [v if v is not None else "" for v in vals]

    # RIR: pad each item to the batch maximum length
    rir_vals = [s.rir for s in samples]
    if all(v is None for v in rir_vals):
        rir_t = None
    else:
        max_len = max(v.shape[-1] for v in rir_vals if v is not None)
        rir_t = torch.stack([
            F.pad(v, (0, max_len - v.shape[-1]))
            for v in rir_vals
        ])

    return ShardBatch(
        speech          = _stack("speech"),
        speech_clean    = _stack("speech_clean"),
        farend          = _stack("farend"),
        echo            = _stack("echo"),
        noise           = _stack("noise"),
        rir             = rir_t,
        rir_onset       = _stack("rir_onset"),
        rir_t60         = _stack("rir_t60"),
        is_real         = _stack("is_real"),
        id_speech       = _str_list("id_speech"),
        id_noise        = _str_list("id_noise"),
        id_farend_echo  = _str_list("id_farend_echo"),
        id_rir          = _str_list("id_rir"),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardSegmentDataset(data.Dataset):
    """
    Fixed-length segment dataset backed by Lhotse shard sources.

    Each __getitem__ call returns a single-sample ShardBatch (no batch
    dimension).  _segment_collate_fn batches them in the DataLoader.

    Parameters
    ----------
    speech_sources:
        One or more ShardSource objects pointing to speech shard directories.
        Add codec.*.tar shards if prob_speech_codec > 0.
    noise_sources:
        Shard sources for noise.  Required if "noise" is in keys.
    farend_echo_sources:
        Shard sources for far-end/echo pairs.  Each cut must have .echo and
        custom["is_real"].  Required if "farend" or "echo" is in keys.
    rir_sources:
        Shard sources for RIRs.  custom dict may contain "onset_sample" and
        "t60".  Required if "rir" is in keys.
    segment_size:
        Output length in samples for speech, noise, farend, and echo.
    length:
        Dataset length — number of __getitem__ calls per epoch.
    sampling_rate:
        Audio sampling rate; used only to convert silence_length to samples.
    shuffle:
        Whether to shuffle shard order within each iterator pass.
    aux_buffer_size:
        Reservoir size for _CyclicCutIterator.  0 = pure sequential order.
    base_seed:
        Base RNG seed.  Worker seeds = base_seed + epoch + worker_id +
        1000 * ddp_rank.
    silence_length:
        Duration (seconds) of silence inserted between concatenated cuts.
    prob_speech_codec:
        Per-item probability of using codec-degraded audio as speech while
        returning the original clean audio as speech_clean.  Requires
        "speech_clean" in keys and codec shards in speech_sources.
    keys:
        Controls which fields are loaded and returned.
    """

    def __init__(
        self,
        speech_sources:       List[ShardSource],
        noise_sources:        Optional[List[ShardSource]] = None,
        farend_echo_sources:  Optional[List[ShardSource]] = None,
        rir_sources:          Optional[List[ShardSource]] = None,
        segment_size:         int   = 48_000,
        length:               int   = 16_384,
        sampling_rate:        int   = 16_000,
        shuffle:              bool  = True,
        aux_buffer_size:      int   = 0,
        base_seed:            int   = 42,
        silence_length:       float = 0.1,
        prob_speech_codec:    float = 0.0,
        keys: tp.List[str]        = ("speech",),
    ):
        super().__init__()

        invalid = set(keys) - VALID_KEYS
        assert not invalid, f"Invalid keys: {invalid}"
        if prob_speech_codec > 0:
            assert "speech_clean" in set(keys), \
                "prob_speech_codec > 0 requires 'speech_clean' in keys"

        self.speech_sources      = speech_sources
        self.noise_sources       = noise_sources or []
        self.farend_echo_sources = farend_echo_sources or []
        self.rir_sources         = rir_sources or []

        self.segment_size      = segment_size
        self.length            = length
        self.sampling_rate     = sampling_rate
        self._shuffle          = shuffle
        self.aux_buffer_size   = aux_buffer_size
        self._silence_len      = max(0, int(silence_length * sampling_rate))
        self.prob_speech_codec = prob_speech_codec

        self.keys: tp.Set[str] = set(keys)

        need_codec = prob_speech_codec > 0
        need_echo  = "echo" in self.keys

        self._speech_fields: List[str] = ["recording"] + (["codec"] if need_codec else [])
        self._noise_fields:  List[str] = ["recording"]
        self._fe_fields:     List[str] = ["recording"] + (["echo"] if need_echo else [])
        self._rir_fields:    List[str] = ["recording"]

        # ── Seed management via shared memory (propagates to persistent workers)
        self._initial_seed = base_seed
        self._epoch_seed   = mp.Value('i', base_seed)

        # ── Auxiliary release via generation counter
        self._release_generation = mp.Value('i', 0)

        # Capture DDP rank pre-fork.
        self._ddp_rank = _get_rank()

        # ── Per-worker lazy state (None until first __getitem__ post-fork)
        self._last_worker_seed: Optional[int]                = None
        self._worker_aux_gen:   int                          = -1
        self._speech_iter:      Optional[_CyclicCutIterator] = None
        self._noise_iter:       Optional[_CyclicCutIterator] = None
        self._fe_iter:          Optional[_CyclicCutIterator] = None
        self._rir_iter:         Optional[_CyclicCutIterator] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Update shared seed for the next epoch.  Call from the main process."""
        with self._epoch_seed.get_lock():
            self._epoch_seed.value = self._initial_seed + epoch

    def release_auxiliary(self) -> None:
        """
        Signal workers to drop their noise/farend-echo/rir iterators on the
        next __getitem__ call.  Call from the main process before validation
        so the aux reservoir buffers are freed while workers are idle.
        """
        with self._release_generation.get_lock():
            self._release_generation.value += 1

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_cyclic(
        self,
        seed:    int,
        offset:  int,
        sources: List[ShardSource],
        fields:  List[str],
    ) -> _CyclicCutIterator:
        return _CyclicCutIterator(
            sources         = sources,
            field_names     = fields,
            base_seed       = seed + offset,
            shuffle         = self._shuffle,
            aux_buffer_size = self.aux_buffer_size,
        )

    def _ensure_iterators(self) -> None:
        info      = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0

        # Release auxiliary iterators if the main process incremented the counter.
        with self._release_generation.get_lock():
            rel_gen = self._release_generation.value
        if rel_gen != self._worker_aux_gen:
            self._noise_iter     = None
            self._fe_iter        = None
            self._rir_iter       = None
            self._worker_aux_gen = rel_gen

        with self._epoch_seed.get_lock():
            epoch_seed = self._epoch_seed.value
        worker_seed = epoch_seed + worker_id + 1_000 * self._ddp_rank

        if worker_seed == self._last_worker_seed and self._speech_iter is not None:
            # Same epoch; rebuild only iterators released by release_auxiliary.
            if self.noise_sources and "noise" in self.keys \
                    and self._noise_iter is None:
                self._noise_iter = self._build_cyclic(
                    worker_seed, 111_111, self.noise_sources, self._noise_fields)
            if self.farend_echo_sources \
                    and ("farend" in self.keys or "echo" in self.keys) \
                    and self._fe_iter is None:
                self._fe_iter = self._build_cyclic(
                    worker_seed, 222_222, self.farend_echo_sources, self._fe_fields)
            if self.rir_sources and "rir" in self.keys \
                    and self._rir_iter is None:
                self._rir_iter = self._build_cyclic(
                    worker_seed, 333_333, self.rir_sources, self._rir_fields)
            return

        # New epoch (or first call): reseed and rebuild all iterators.
        random.seed(worker_seed)
        np.random.seed(worker_seed)

        self._speech_iter = self._build_cyclic(
            worker_seed, 0, self.speech_sources, self._speech_fields)

        self._noise_iter = self._build_cyclic(
            worker_seed, 111_111, self.noise_sources, self._noise_fields) \
            if self.noise_sources and "noise" in self.keys else None

        self._fe_iter = self._build_cyclic(
            worker_seed, 222_222, self.farend_echo_sources, self._fe_fields) \
            if self.farend_echo_sources \
               and ("farend" in self.keys or "echo" in self.keys) else None

        self._rir_iter = self._build_cyclic(
            worker_seed, 333_333, self.rir_sources, self._rir_fields) \
            if self.rir_sources and "rir" in self.keys else None

        self._last_worker_seed = worker_seed

    # ── Audio loading helpers ─────────────────────────────────────────────────

    def _gen_audio(
        self,
        cut_iter:   _CyclicCutIterator,
        echo_field: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], bool]:
        """
        Pull cuts from cut_iter, concatenating with silence until segment_size.
        If echo_field is True, also loads cut.echo at the same crop offset so
        farend and echo remain temporally aligned.
        Returns (primary, echo_or_None, cut_ids, is_real_from_first_cut).
        """
        primary_chunks: List[np.ndarray] = []
        echo_chunks:    List[np.ndarray] = []
        ids:            List[str]        = []
        is_real:        bool             = False
        remaining:      int              = self.segment_size

        while remaining > 0:
            cut = cut_iter.next()
            if not ids:
                is_real = bool((cut.custom or {}).get("is_real", False))
            ids.append(cut.id)

            primary = _to_numpy(cut.load_audio())
            echo    = _to_numpy(cut.echo.load_audio()) if echo_field else None

            T = primary.shape[-1]
            if remaining >= T:
                primary_chunks.append(primary)
                if echo is not None:
                    echo_chunks.append(echo)
                remaining -= T
                if remaining > 0 and self._silence_len > 0:
                    sil = min(remaining, self._silence_len)
                    primary_chunks.append(np.zeros((1, sil), dtype=np.float32))
                    if echo is not None:
                        echo_chunks.append(np.zeros((1, sil), dtype=np.float32))
                    remaining -= sil
            else:
                start = random.randint(0, T - remaining)
                primary_chunks.append(primary[:, start : start + remaining])
                if echo is not None:
                    echo_chunks.append(echo[:, start : start + remaining])
                remaining = 0

        primary_out = np.concatenate(primary_chunks, axis=-1)
        echo_out    = np.concatenate(echo_chunks, axis=-1) if echo_chunks else None
        return primary_out, echo_out, ids, is_real

    def _gen_speech(
        self,
        cut_iter:   _CyclicCutIterator,
        load_codec: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Like _gen_audio but co-loads codec audio at the same crop offset.
        Returns (speech, speech_clean, cut_ids).
        When load_codec is False, speech and speech_clean are the same array.
        """
        speech_chunks: List[np.ndarray] = []
        clean_chunks:  List[np.ndarray] = []
        ids:           List[str]        = []
        remaining:     int              = self.segment_size

        while remaining > 0:
            cut = cut_iter.next()
            ids.append(cut.id)

            clean  = _to_numpy(cut.load_audio())
            speech = _to_numpy(cut.codec.load_audio()) if load_codec else clean

            T = clean.shape[-1]
            if remaining >= T:
                speech_chunks.append(speech)
                clean_chunks.append(clean)
                remaining -= T
                if remaining > 0 and self._silence_len > 0:
                    sil = min(remaining, self._silence_len)
                    speech_chunks.append(np.zeros((1, sil), dtype=np.float32))
                    clean_chunks.append(np.zeros((1, sil), dtype=np.float32))
                    remaining -= sil
            else:
                start = random.randint(0, T - remaining)
                speech_chunks.append(speech[:, start : start + remaining])
                clean_chunks.append(clean[:, start : start + remaining])
                remaining = 0

        return (
            np.concatenate(speech_chunks, axis=-1),
            np.concatenate(clean_chunks,  axis=-1),
            ids,
        )

    def _load_rir(
        self,
        cut_iter: _CyclicCutIterator,
    ) -> Tuple[np.ndarray, int, float, str]:
        """Load one RIR cut at its native length.  Returns (rir, onset, t60, id)."""
        cut   = cut_iter.next()
        rir   = _to_numpy(cut.load_audio())
        onset = int((cut.custom or {}).get("onset_sample", 0))
        t60   = float((cut.custom or {}).get("t60", -1.0))
        return rir, onset, t60, cut.id

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> ShardBatch:  # idx ignored; iterators drive order
        self._ensure_iterators()

        # ── Speech (+ optional codec) ──────────────────────────────────────────
        load_codec = (
            "speech_clean" in self.keys
            and random.random() < self.prob_speech_codec
        )
        speech, speech_clean, id_speech = self._gen_speech(self._speech_iter, load_codec)

        # ── Noise ──────────────────────────────────────────────────────────────
        noise:       Optional[np.ndarray] = None
        id_noise:    Optional[str]        = None
        if "noise" in self.keys and self._noise_iter is not None:
            noise_arr, _, id_noise_list, _ = self._gen_audio(self._noise_iter)
            noise    = noise_arr
            id_noise = "|".join(id_noise_list)

        # ── Farend + Echo ──────────────────────────────────────────────────────
        farend:    Optional[np.ndarray] = None
        echo:      Optional[np.ndarray] = None
        is_real:   bool                 = False
        id_fe:     Optional[str]        = None
        if ("farend" in self.keys or "echo" in self.keys) and self._fe_iter is not None:
            farend_arr, echo_arr, id_fe_list, is_real = self._gen_audio(
                self._fe_iter, echo_field="echo" in self.keys
            )
            if "farend" in self.keys:
                farend = farend_arr
            if "echo" in self.keys:
                echo = echo_arr
            id_fe = "|".join(id_fe_list)

        # ── RIR ────────────────────────────────────────────────────────────────
        rir:       Optional[np.ndarray] = None
        rir_onset: Optional[int]        = None
        rir_t60:   Optional[float]      = None
        id_rir:    Optional[str]        = None
        if "rir" in self.keys and self._rir_iter is not None:
            rir, rir_onset, rir_t60, id_rir = self._load_rir(self._rir_iter)

        return ShardBatch(
            speech       = torch.from_numpy(speech).float(),
            speech_clean = torch.from_numpy(speech_clean).float()
                           if "speech_clean" in self.keys else None,
            noise        = torch.from_numpy(noise).float()
                           if noise is not None else None,
            farend       = torch.from_numpy(farend).float()
                           if farend is not None else None,
            echo         = torch.from_numpy(echo).float()
                           if echo is not None else None,
            rir          = torch.from_numpy(rir).float()
                           if rir is not None else None,
            rir_onset    = torch.tensor(rir_onset, dtype=torch.long)
                           if rir_onset is not None and "rir_onset" in self.keys else None,
            rir_t60      = torch.tensor(rir_t60, dtype=torch.float32)
                           if rir_t60 is not None and "rir_t60" in self.keys else None,
            is_real      = torch.tensor(is_real, dtype=torch.bool)
                           if "is_real" in self.keys else None,
            id_speech    = "|".join(id_speech) if "id_speech" in self.keys else None,
            id_noise     = id_noise            if "id_noise" in self.keys else None,
            id_farend_echo = id_fe             if "id_farend_echo" in self.keys else None,
            id_rir       = id_rir              if "id_rir" in self.keys else None,
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def build_shard_segment_dataloader_from_hps(
    hparams,
    mode: str,
    keys: tp.List[str] = ("speech",),
) -> Tuple[ShardSegmentDataset, torch.utils.data.DataLoader]:
    """HParams-driven factory; called from utils/data/__init__.py."""
    hp = hparams.data[mode]

    speech_sources = [ShardSource(**kw) for kw in hp.inputs.speech]

    noise_sources: Optional[List[ShardSource]] = None
    if hp.inputs.get("noise"):
        noise_sources = [ShardSource(**kw) for kw in hp.inputs.noise]

    farend_echo_sources: Optional[List[ShardSource]] = None
    if hp.inputs.get("farend_echo"):
        farend_echo_sources = [ShardSource(**kw) for kw in hp.inputs.farend_echo]

    rir_sources: Optional[List[ShardSource]] = None
    if hp.inputs.get("rir"):
        rir_sources = [ShardSource(**kw) for kw in hp.inputs.rir]

    return build_shard_segment_dataloader(
        speech_sources      = speech_sources,
        noise_sources       = noise_sources,
        farend_echo_sources = farend_echo_sources,
        rir_sources         = rir_sources,
        segment_size        = hp.get("segment_size", 48_000),
        length              = hp.get("length", 16_384),
        batch_size          = hp.get("batch_size", 32),
        sampling_rate       = hparams.data.sampling_rate,
        shuffle             = hp.get("shuffle", True),
        aux_buffer_size     = hp.get("aux_buffer_size", 0),
        base_seed           = hparams.train.seed,
        silence_length      = hp.get("silence_length", 0.1),
        prob_speech_codec   = hp.get("prob_speech_codec", 0.0),
        keys                = keys,
        num_workers         = hp.get("num_workers", 0),
        pin_memory          = hp.get("pin_memory", False),
        persistent_workers  = hp.get("persistent_workers", False),
        prefetch_factor     = hp.get("prefetch_factor", 2),
    )


def build_shard_segment_dataloader(
    speech_sources:       List[ShardSource],
    noise_sources:        Optional[List[ShardSource]] = None,
    farend_echo_sources:  Optional[List[ShardSource]] = None,
    rir_sources:          Optional[List[ShardSource]] = None,
    segment_size:         int   = 48_000,
    length:               int   = 16_384,
    batch_size:           int   = 32,
    sampling_rate:        int   = 16_000,
    shuffle:              bool  = True,
    aux_buffer_size:      int   = 0,
    base_seed:            int   = 42,
    silence_length:       float = 0.1,
    prob_speech_codec:    float = 0.0,
    keys: tp.List[str]        = ("speech",),
    num_workers:          int   = 0,
    pin_memory:           bool  = False,
    persistent_workers:   bool  = False,
    prefetch_factor:      int   = 2,
) -> Tuple[ShardSegmentDataset, torch.utils.data.DataLoader]:
    """Build and return (dataset, dataloader)."""
    dataset = ShardSegmentDataset(
        speech_sources      = speech_sources,
        noise_sources       = noise_sources,
        farend_echo_sources = farend_echo_sources,
        rir_sources         = rir_sources,
        segment_size        = segment_size,
        length              = length,
        sampling_rate       = sampling_rate,
        shuffle             = shuffle,
        aux_buffer_size     = aux_buffer_size,
        base_seed           = base_seed,
        silence_length      = silence_length,
        prob_speech_codec   = prob_speech_codec,
        keys                = keys,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        collate_fn         = _segment_collate_fn,
        pin_memory         = pin_memory if num_workers > 0 else False,
        persistent_workers = persistent_workers if num_workers > 0 else False,
        prefetch_factor    = prefetch_factor if num_workers > 0 else None,
        drop_last          = True,
    )
    return dataset, loader
