from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import Dict, Tuple, Optional, Union, List, Any

import torch
from torchaudio.transforms import Resample
from torch import Tensor
import torch.distributed as dist
import numpy as np
from pesq import pesq as measure_pesq
from pystoi import stoi as measure_stoi
from utils.terminal import clear_current_line


SAMPLING_RATE = {
    "pesq": 16_000,
    "stoi": 10_000,
}


class Metrics:
    def __init__(
        self,
        num_workers: int,
        sr: int,
        world_size: int,
        rank: int,
        device,
        pesq: bool = True,
        stoi: bool = True,
    ) -> None:
        self.num_workers = num_workers
        self.sr = sr
        self.world_size = world_size
        self.rank = rank
        self.device = device

        self.states = {}
        self.resamplers = {}
        for name in ["pesq", "stoi"]:
            if eval(name):
                sr = SAMPLING_RATE[name]
                if sr not in self.resamplers:
                    self.resamplers[sr] = Resample(
                        orig_freq=self.sr, new_freq=sr
                    ).to(device)

                # On some servers, STOI never finishes if used with multi-processing.
                # Therefore, we use single-processing for STOI.
                self.states[name] = {
                    "sr": sr,
                    "mean": 0.0,
                    "best": 0.0,
                    "futures": [],
                }
        self.executor = None
        self.num_items = 0

    def initialize(self) -> None:
        for name in self.states:
            self.states[name]["futures"] = []
            self.states[name]["mean"] = 0.0
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self.num_items = 0

    def submit(
        self,
        ref: Tensor,
        deg: Tensor,
        wav_lens: Optional[Union[Tensor, List[Any]]] = None
    ) -> None:
        """ref/deg: [B, 1, T] or [B, T]"""
        assert self.executor is not None
        batch_size = ref.size(0)
        deg = deg.to(dtype=torch.float32)
        if ref.ndim == 3:
            ref = ref.squeeze(1)
        if deg.ndim == 3:
            deg = deg.squeeze(1)

        cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for name, state in self.states.items():
            sr = state["sr"]
            if sr not in cache:
                ref_np: np.ndarray = self.resamplers[sr](ref).cpu().numpy()
                deg_np: np.ndarray = self.resamplers[sr](deg).cpu().numpy()
                cache[sr] = (ref_np, deg_np)

        for i in range(batch_size):
            file_idx = self.world_size * (self.num_items + i) + self.rank
            for name, state in self.states.items():
                sr = state["sr"]
                ref_np, deg_np = cache[sr]
                ref_i = ref_np[i]
                deg_i = deg_np[i]
                if wav_lens is not None:
                    wav_len = int(wav_lens[i] * sr / self.sr)
                    ref_i = ref_i[:wav_len]
                    deg_i = deg_i[:wav_len]
                if name == "pesq":
                    future = self.executor.submit(measure_pesq, 16000, ref_i, deg_i, "wb")
                    state["futures"].append(future)
                elif name == "stoi":
                    state["mean"] += measure_stoi(ref_i, deg_i, 10000)
        self.num_items += batch_size

    def retrieve(self, verbose: bool) -> Dict[str, float]:
        padding = int(math.log10(self.num_items)) + 1
        for name, state in self.states.items():
            if name == "stoi":
                continue
            else:
                for idx, future in enumerate(as_completed(state["futures"]), start=1):
                    state["mean"] += future.result()
                    if verbose:
                        print(
                            f"\r{name.upper()} {idx:{padding}d}/{self.num_items}"
                            f"    score: {state['mean'] / idx:6.4f}",
                            end='', flush=True
                        )
        if verbose:
            clear_current_line()
            print("\rwaiting for ProcessPoolExecutor to shutdown...", end="", flush=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        if verbose:
            clear_current_line()

        metric_mean = torch.tensor(
            [state["mean"] for state in self.states.values()],
            device=self.device
        )
        dist.reduce(metric_mean, dst=0, op=dist.ReduceOp.SUM)
        self.num_items *= self.world_size
        for idx, state in enumerate(self.states.values()):
            state["mean"] = metric_mean[idx].item() / self.num_items
            if state["best"] < state["mean"]:
                state["best"] = state["mean"]
        metrics = {
            f"metrics/{name}": state["mean"] for name, state in self.states.items()
        }
        metrics.update({
            f"metrics/best_{name}": state["best"] for name, state in self.states.items()
        })
        return metrics

    def state_dict(self) -> Dict[str, float]:
        return {
            f"best_{name}": state["best"] for name, state in self.states.items()
        }

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        try:
            for name, value in state_dict.items():
                name = name[len("best_"):]  # e.g. "best_pesq" -> "pesq"
                self.states[name]["best"] = value
        except Exception as e:
            print(e)
