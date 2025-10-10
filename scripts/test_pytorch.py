from pathlib import Path
import argparse
import os

import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from utils import get_hparams
from wrappers import get_wrapper


def main(args):
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load a model
    device = torch.device("cuda")
    base_dir = os.path.join('logs', args.name)
    hps = get_hparams(base_dir=base_dir)
    wrapper = get_wrapper(hps.wrapper)(hps, device=device)
    wrapper.load()

    filelists = list(Path(args.input_dir).glob('*.wav'))
    for noisy_path in tqdm(filelists, dynamic_ncols=True, smoothing=0.0):
        # Load a noisy audio
        noisy, fs = librosa.load(noisy_path, sr=16000, mono=True)
        noisy = torch.from_numpy(noisy).float().to(device).unsqueeze(0)

        # Inference
        with torch.no_grad():
            enhanced, _ = wrapper.model(noisy)  # return: wav, spec

        # Save the enhanced audio
        sf.write(args.output_dir / noisy_path.name, enhanced.squeeze().cpu().numpy(), fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test model')
    parser.add_argument(
        '-n', '--name',
        type=str,
        required=True,
        help='The latest checkpoint in logs/{name} will be loaded.'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='/home/shahn/Datasets/DNS-Challenge/16khz/testset_synthetic_interspeech2020/no_reverb/noisy',
        help='The dir path including noisy wavs for evaluation.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='enhanced/dns',
        help='The dir path to save enhanced wavs.'
    )

    args = parser.parse_args()
    main(args)
