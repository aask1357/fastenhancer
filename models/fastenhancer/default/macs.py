if __name__ == "__main__":
    from utils import get_hparams
    hp = get_hparams("configs/rnnformer/size/t_bn.yaml")["model_kwargs"]
    SR = 16000
    F_INPUT = hp.n_fft // 2
    H = hp.hop_size
    T = SR / H

    C1 = 24 #hp.channels
    C2 = 20 #hp.rnnformer_kwargs.channels
    F1 = F_INPUT // hp.stride
    F2 = 16 #hp.rnnformer_kwargs.freq
    K = 1 #hp.rnnformer_kwargs.num_blocks
    NH = hp.rnnformer_kwargs.num_heads

    # Pre Encoder
    k = hp.kernel_size[0]
    num_macs = 2 * C1 * k * F1 * T     # conv(2, C1, k=8, s=4)
    print(f"PreEncoder: {num_macs/1000_000:.1f}M")
    num_macs_total = num_macs

    # Encoder Blocks
    num_macs = 0
    for k in hp.kernel_size[1:]:
        num_macs += C1**2 * k * F1 * T
    print(f"Encoder Blocks: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Pre RNNFormer
    num_macs = (
        F1 * F2 * C1    # linear(F1 -> F2)
        + C1 * C2 * F2  # conv(C1 -> C2)
    ) * T
    print(f"Pre RNNFormer: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # RNNFormer
    num_macs = 0
    for i in range(K):
        # RNN
        num_macs += C2**2 * 6 * F2 * T
        
        # RNN FC
        num_macs += C2 * C2 * F2 * T
        
        # MHSA
        num_macs += (
            C2 * C2 * 3 * F2    # QKV = linear(x)
            + F2 * C2 * F2      # attn_map = Q @ K^T -> NH * F2 * C2/NH * F2
            + F2 * F2 * C2      # out = attn_map @ V -> NH * F2 * F2 * C2/NH
        ) * T
        
        # MHSA FC
        num_macs += C2 * C2 * F2 * T
    print(f"RNNFormer: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Post RNNFormer
    num_macs = (
        F2 * F1 * C2    # linear(F2 -> F1)
        + C2 * C1 * F1  # conv(C2 -> C1)
    ) * T
    print(f"Post RNNFormer: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Decoder Blocks
    num_macs = 0
    for idx in range(len(hp.kernel_size)-1, 0, -1):
        k = hp.kernel_size[idx]
        num_macs += (
            2 * C1**2 * F1      # conv(2*C1, C1, 1)
            + C1**2 * k * F1    # conv(C1, C1, k)
        ) * T
    print(f"Decoder Blocks: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Post Decoder
    k = hp.kernel_size[0]
    num_macs = (
        2 * C1**2 * F1      # conv(2*C1, C1, 1)
        + C1 * 2 * k * F1   # conv_transpsoe(C1, 2, k=8, s=4)
    ) * T
    print(f"Post Decoder: {num_macs/1000_000:.2f}M")
    num_macs_total += num_macs

    print(f"Total MACs: {num_macs_total/1000_000:.2f}M")
