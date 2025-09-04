# Copyright (c) 2024 Alessandro Ragano
# MIT license
# Code from https://github.com/alessandroragano/scoreq/tree/main

import os
from typing import Optional
from urllib.request import urlretrieve

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


# The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
PADDING_MULTIPLE = 320

def dynamic_pad(x: np.ndarray) -> np.ndarray:
    """Pads the input tensor to be a multiple of PADDING_MULTIPLE."""
    length = x.shape[-1]
    required_len = (length + PADDING_MULTIPLE - 1) // PADDING_MULTIPLE * PADDING_MULTIPLE
    remainder = required_len - length
    return np.pad(x, ((0, 0), (0, remainder)))


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(n - self.n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Scoreq():
    """
    Main class for handling the SCOREQ audio quality assessment model.
    Defaults to using high-performance ONNX models.
    """
    def __init__(self, data_domain='natural', mode='nr', device='cpu', num_threads: Optional[int] = None):
        """
        Initializes the Scoreq object.

        Args:
            data_domain (str): Domain of audio ('natural' or 'synthetic').
            mode (str): Mode of operation ('nr' or 'ref').
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.data_domain = data_domain
        self.mode = mode
        self.session = None
        self.device = device
        self.num_threads = num_threads

        self.init_onnx()

    def init_onnx(self):
        """Initializes the ONNX Runtime session."""
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif self.device.startswith('cuda'):
            providers = ['CUDAExecutionProvider']
        else:
            raise ValueError(f"Unsupported device: {self.device}. Use 'cpu' or 'cuda:n'.")

        domain_part = 'telephone' if self.data_domain == 'natural' else 'synthetic'
        mode_part = 'adapt_nr' if self.mode == 'nr' else 'fixed_nmr'
        onnx_filename = f"{mode_part}_{domain_part}.onnx"

        ZENODO_ONNX_URLS = {
            'adapt_nr_telephone.onnx': 'https://zenodo.org/records/15739280/files/adapt_nr_telephone.onnx',
            'fixed_nmr_telephone.onnx': 'https://zenodo.org/records/15739280/files/fixed_nmr_telephone.onnx',
            'adapt_nr_synthetic.onnx': 'https://zenodo.org/records/15739280/files/adapt_nr_synthetic.onnx',
            'fixed_nmr_synthetic.onnx': 'https://zenodo.org/records/15739280/files/fixed_nmr_synthetic.onnx',
        }

        model_url = ZENODO_ONNX_URLS.get(onnx_filename)
        if not model_url:
            raise ValueError(f"Invalid model combination: domain='{self.data_domain}', mode='{self.mode}'")

        model_path = self._download_model(onnx_filename, model_url, cache_dir_name="onnx-models")

        opts = ort.SessionOptions()
        if self.num_threads is not None:
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
        self.device = self.session.get_providers()[0]
        print(f"SCOREQ (ONNX) initialized on provider: {self.device}")

    def _download_model(self, filename, url, cache_dir_name):
        """Helper to download a model from a URL with a progress bar."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scoreq", cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, filename)

        if not os.path.exists(model_path):
            print(f"Downloading {filename}...")
            try:
                with TqdmUpTo(
                    unit='B',
                    unit_scale=True,
                    miniters=1,
                    desc=filename,
                    dynamic_ncols=True
                ) as t:
                    urlretrieve(url, model_path, reporthook=t.update_to)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                if os.path.exists(model_path): os.remove(model_path)
                raise e

        return model_path

    def __call__(self, deg: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if deg.ndim == 1:
            deg = deg[np.newaxis, :]
            ref = ref[np.newaxis, :]
        elif deg.ndim == 3:
            deg = deg.squeeze(1)
            ref = ref.squeeze(1)
        assert deg.ndim == 2 and ref.ndim == 2, \
            f"Input tensors must be 2D arrays, got {deg.ndim}D (deg) and {ref.ndim}D (ref)."

        input_name = self.session.get_inputs()[0].name
        deg = dynamic_pad(deg)
        if self.mode == 'nr':
            score = self.session.run(None, {input_name: deg})[0]
        elif self.mode == 'ref':
            ref = dynamic_pad(ref)

            deg_emb = self.session.run(None, {input_name: deg})[0]  # [Batch, Channels]
            ref_emb = self.session.run(None, {input_name: ref})[0]  # [Batch, Channels]
            score = np.linalg.norm(deg_emb - ref_emb, axis=1)       # [Batch]
        else:
            raise ValueError("Invalid mode specified.")

        return score


if __name__ == "__main__":
    # Example usage
    scoreq = Scoreq(data_domain='natural', mode='ref', device='cuda')
    ref_audio = np.random.randn(3, 1, 16000).astype(np.float32)  # Simulated reference audio
    deg_audio = ref_audio + np.random.randn(3, 1, 16000).astype(np.float32) * 0.05  # Simulated degraded audio

    score = scoreq(deg_audio, ref_audio)
    print(f"SCOREQ Score: {score}")
