if __name__ == "__main__":
    from utils import get_hparams
    hp = get_hparams("configs/se/bsrnn.yaml")["model_kwargs"]
    SR = 16000
    F_INPUT = hp.n_fft // 2
    H = hp.hop_size
    T = SR / H

    C = hp.num_channels
    H = 2 * C
    L = hp.num_layers
    S = [
        2,    3,    3,    3,    3,   3,   3,    3,    3,    3,   3,
        8,    8,    8,    8,    8,   8,   8,    8,    8,    8,   8,   8,
        16,   16,   16,   16,   16,  16,  16,   17
    ]   # sum(S) == 257

    # Band Split
    num_macs = 0
    num_params = 0
    for s in S:
        num_macs += 2*s * C * T
        num_params += 2*s * C + C
    print(f"BandSplit - MACs: {num_macs/1000_000:.1f}M, #Params: {num_params/1000:.1f}K")
    num_macs_total = num_macs
    num_params_total = num_params

    # RNN
    num_macs = (
        C*H * 4 + H**2 * 4 + H * C              # RNN-Time + FC-Time
        + (C*H * 4 + H**2 * 4) * 2 + 2*H * C    # RNN-Freq + FC-Freq
    ) * len(S) * T * L
    num_params =  (
        C*H * 4 + H**2 * 4 + 8*H + H * C + C
        + (C*H * 4 + H**2 * 4 + 8*H) * 2 + 2*H * C + C
    ) * L
    print(f"RNN - MACs: {num_macs/1000_000:.1f}M, #Params: {num_params/1000:.1f}K")
    num_macs_total += num_macs
    num_params_total += num_params

    # Mask Decoder
    num_macs = 0
    num_params = 0
    for s in S:
        num_macs += (C * C*4 + 4*C * 4*s) * T * 2
        num_params += (C * C*4 + 4*C + 4*C * 4*s + 4*s) * 2
    print(f"Mask Decoder - MACs: {num_macs/1000_000:.1f}M, #Params: {num_params/1000:.1f}K")
    num_macs_total += num_macs
    num_params_total += num_params

    print(f"Total - MACs: {num_macs_total/1000_000:.2f}M, #Params: {num_params_total/1000:.2f}K")
