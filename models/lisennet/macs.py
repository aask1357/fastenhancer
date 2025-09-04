from utils import get_hparams


def calc_gru(input_size, hidden_size):
    return (input_size+hidden_size)*hidden_size*3 + hidden_size*3


if __name__ == "__main__":
    hps = get_hparams("configs/se/lisennet.yaml")

    hp = hps["model_kwargs"]
    C = hp.num_channels
    N = hp.n_blocks
    F1 = hp.n_fft // 2 + 1
    H = hp.hop_size
    SR = 16_000
    T = SR / hp.hop_size

    # Encoder
    num_macs = 3 * C//4 * F1 * T   # conv1
    c_in, f = C//4, F1
    for c_out, f in zip((C//2, C//4*3, C), (257, 128, 64)):
        f_out_high = (f - f // 4 + 2 - 5) // 3 + 1  # = (f_in + 2*padding - kernel_size) // stride + 1
        num_macs += (
            2 * 3 * f//4            # low conv
            + 2 * 5 * f_out_high    # high conv
        ) * c_out * c_out * T
        c_in = c_out
    print(f"Encoder: {num_macs/1000_000:.1f}M")
    num_macs_total = num_macs

    # DPRNN
    h, f = 24, 32
    num_macs = 0
    for _ in range(N):
        num_macs += (
            calc_gru(C, h//2) * 2 + h * C   # IntraRNN
            + calc_gru(C, h) + h * C        # InterRNN
        ) * f * T
        num_macs += (   # ConvGLU
            C * C*4     # fc1
            + C*2 * 3   # dwconv
            + C*2       # act(dwconv) * v
            + C*2 * C   # fc2
        ) * f * T
    print(f"DPRNN: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Decoder
    num_macs = 0
    c_in, f = C, 32
    for c_out in (C // 4 * 3, C // 2, C // 4):
        num_macs += (
            3 * f//2        # low conv
            + 3 * 3 * f//2  # high conv
        ) * c_in*2 * c_out * T
        c_in = c_out
        f = f * 2
    f += 1
    num_macs += (
        c_out * 2 * 2*2     # mask_conv.0
        + 2 * 2 * 1         # mask_conv.3
        + 2 * 2             # LSigmoid
    ) * f * T
    print(f"Decoder: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    print(f"Total MACs: {num_macs_total/1000_000:.2f}M")
