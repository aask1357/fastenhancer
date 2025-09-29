import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    # Set the device & number of threads
    if args.device == "cuda":
        args.device = "cuda:0"  # DNSMOS requires a specific GPU device
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    # Import libraries after setting the num_threads
    import torch
    from torchaudio.transforms import Resample
    import torchmetrics
    from pystoi import stoi
    from pesq import pesq

    from functional import get_mask
    from utils import get_hparams, HParams
    from utils.data import get_dataset_dataloader
    from utils.scoreq_onnx import Scoreq
    from wrappers import get_wrapper

    torch.set_num_threads(args.num_threads)

    def product(s1, s2):
        norm = torch.sum(s1*s2, -1, keepdim=True)
        return norm

    def si_snr(s1, s2, mask, eps: float = 1e-7):
        # s1: wav_hat / s2: wav
        s1_s2_norm = product(s1, s2)
        s2_s2_norm = product(s2, s2)
        s_target =  s1_s2_norm / (s2_s2_norm + eps) * s2
        e_nosie = s1 - s_target
        target_norm = product(s_target, s_target)
        noise_norm = product(e_nosie, e_nosie)
        snr = torch.log10((target_norm) / (noise_norm + eps) + eps)
        return 10.0 * torch.sum(snr * mask, dim=1) / mask.sum(dim=1)

    # Load the model
    base_dir = os.path.join("logs", args.name)
    hps = get_hparams(f"{base_dir}/config.yaml", base_dir)
    wrapper = get_wrapper(hps.wrapper)(hps, device=args.device)
    wrapper.load(epoch=args.epoch)
    wrapper.eval()
    # wrapper.remove_weight_reparameterizations()

    sr = hps.data.sampling_rate
    hop_size = wrapper.hop_size

    def print_num_params(module, prefix=""):
        n_params = 0
        for n, p in module.named_parameters():
            n_params += p.numel()
        print(f"{prefix}#params: {n_params/1000} K")

    print_num_params(wrapper.model)

    # Load the dataset
    hps.pesq.batch_size = args.batch_size
    hps.pesq.num_workers = 0
    hps.data.pesq.transcript_dir = args.transcript_dir
    keys = ["clean", "noisy", "wav_len"]
    if args.wer:
        keys.append("transcript")
    _, dataloader = get_dataset_dataloader(
        hps,
        mode="pesq",
        keys=keys,
    )

    # Prepare DNSMOS, Resampler
    dnsmos = torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore(
        fs=16_000,
        personalized=False,
        device=args.device,
        num_threads=args.num_threads,
    )
    resampler10khz = Resample(sr, 10000).to(args.device)
    resampler16khz = Resample(sr, 16000).to(args.device)

    # Prepare SCOREQ
    scoreq = Scoreq(
        data_domain='natural',
        mode='ref',
        device=args.device,
        num_threads=args.num_threads,
    )

    # Prepare an ASR model and text cleaner for WER calculation
    if args.wer:
        import whisper
        import jiwer
        from whisper.normalizers import EnglishTextNormalizer
        model = whisper.load_model("turbo").to(args.device)
        options = whisper.DecodingOptions(language="en", without_timestamps=True)
        normalizer = EnglishTextNormalizer()

    # Calculate metrics
    dnsmos_total, scoreq_total, sisdr_total, pesq_total, stoi_total, estoi_total = 0, 0, 0, 0, 0, 0
    wer_total = 0
    num_total = 0

    for idx, batch in enumerate(dataloader, start=1):
        # Load batch
        if args.wer:
            text_ref = batch["transcript"]
        wav_clean = batch["clean"].to(args.device)
        wav_noisy = batch["noisy"].to(args.device)
        wav_lens = batch["wav_len"].to(args.device) // hop_size * hop_size
        mask = get_mask(wav_lens).squeeze(1)

        batch_wav_len = wav_clean.size(-1) // hop_size * hop_size
        wav_clean = wav_clean[..., :batch_wav_len] * mask
        wav_noisy = wav_noisy[..., :batch_wav_len] * mask

        # Forward
        with torch.no_grad():
            wav_clean_hat, _ = wrapper.model(wav_noisy)
        wav_clean_hat = wav_clean_hat * mask

        # SISDR
        result = si_snr(wav_clean_hat, wav_clean, mask)
        sisdr_total += result.sum()

        # Resampling
        wav_clean_10khz = resampler10khz(wav_clean).cpu().numpy()
        wav_clean_hat_10khz = resampler10khz(wav_clean_hat).cpu().numpy()
        wav_clean_16khz = resampler16khz(wav_clean).cpu().numpy()
        wav_clean_hat_16khz_tensor = resampler16khz(wav_clean_hat)
        wav_clean_hat_16khz = wav_clean_hat_16khz_tensor.cpu().numpy()

        for i in range(len(wav_lens)):
            # PESQ
            wav_len_16khz = wav_lens[i] * 16000 // sr
            ref = wav_clean_16khz[i, :wav_len_16khz]
            deg = wav_clean_hat_16khz[i, :wav_len_16khz]
            result = pesq(16000, ref, deg, "wb")
            pesq_total += result

            # WER
            deg = wav_clean_hat_16khz_tensor[i, :wav_len_16khz]
            if args.wer:
                audio = whisper.pad_or_trim(deg.flatten())
                mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
                out = model.decode(mel, options)
                text_pred = out.text
                wer_total += jiwer.wer(
                    normalizer(text_ref[i]),
                    normalizer(text_pred),
                ) * 100  # Convert to percentage

            # DNSMOS
            result = dnsmos(deg)
            dnsmos_total += result

            # SCOREQ
            deg = wav_clean_hat_16khz[i, :wav_len_16khz]
            ref = wav_clean_16khz[i, :wav_len_16khz]
            scoreq_total += scoreq(deg, ref).item()

            # STOI
            wav_len_10khz = wav_lens[i] * 10000 // sr
            ref = wav_clean_10khz[i, :wav_len_10khz]
            deg = wav_clean_hat_10khz[i, :wav_len_10khz]
            result = stoi(ref, deg, 10000, extended=False)
            stoi_total += result

            # ESTOI
            result = stoi(ref, deg, 10000, extended=True)
            estoi_total += result

            # print
            num_total += 1
            print(
                f"\r({num_total}/{len(dataloader.dataset)}) "
                "[p808_mos, sig, bak, ovr, scoreq, sisdr, pesq, stoi, estoi, wer]: "
                f"{dnsmos_total[0] / num_total:.2f}, "
                f"{dnsmos_total[1] / num_total:.2f}, "
                f"{dnsmos_total[2] / num_total:.2f}, "
                f"{dnsmos_total[3] / num_total:.2f}, "
                f"{scoreq_total / num_total:.3f}, "
                f"{sisdr_total / num_total:.1f}, "
                f"{pesq_total / num_total:.2f}, "
                f"{stoi_total / num_total:.3f}, "
                f"{estoi_total / num_total:.3f}, "
                f"{wer_total / num_total:>3.1f}",
                end="", flush=True
            )
    out = f"\n{args.name}: "
    for score in (
        dnsmos_total[0], dnsmos_total[1], dnsmos_total[2],
        dnsmos_total[3], scoreq_total, sisdr_total, pesq_total,
        stoi_total, estoi_total,
    ):
        out = f"{out}{score / num_total:.6f}, "
    print(f"{out}{wer_total / num_total:>3.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help="Path of the model checkpoint",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda:0",
        help="Device for running ASR inference. cpu | cuda:0. Default: cuda:0",
    )
    parser.add_argument(
        "-e", "--epoch",
        type=int,
        help="Epoch of the model checkpoint",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default="/home/shahn/Datasets/voicebank-demand/logfiles/transcript_testset.txt",
        help="Path to the transcript file for the testset. Not used if --wer is False.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for running models. Recommend to increase this when running on CPU.",
    )
    parser.add_argument(
        "--wer",
        type=str2bool,
        default=True,
        help="Whether to calculate WER. Default: True"
    )
    args = parser.parse_args()

    main(args)
