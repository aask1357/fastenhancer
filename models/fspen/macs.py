from utils import get_hparams
from models import get_wrapper


CONV_OUTPUT_LENGTH = True
CONVT_INPUT_LENGTH = True
COUNT_BN = 0
COUNT_LN = 0
GRU_THOP_VER = False
COUNT_BIAS = False


def calc_gru(input_size, hidden_size):
    if GRU_THOP_VER:
        return (input_size+hidden_size)*hidden_size*3 + hidden_size*12
    else:
        return (input_size+hidden_size)*hidden_size*3 + hidden_size*3


if __name__ == "__main__":
    hps = get_hparams("configs/se/fspen.yaml")
    # wrapper = get_wrapper(hps.model)(hps, train=False)
    # import torch
    # x = torch.randn(1, 257, 130, 2)
    # with torch.profiler.profile(
    #     with_flops=True
    # ) as p:
    #     y = wrapper.model.model_forward(x)
    # print(p.key_averages())
    # exit()
    
    hp = hps["model_kwargs"]
    C1 = hp.channels
    K = hp.kernel_size
    S = hp.stride
    F1 = hp.n_fft // 2 + 1
    C2 = hp.dpe_kwargs.channels
    SR = 16_000
    T = SR / hp.hop_size

    # Full-band encoder
    F = F1
    num_macs = 0
    for idx in range(len(K)):
        if CONV_OUTPUT_LENGTH:
            F = F // S[idx]
        Cin = 2 if idx == 0 else C1[idx-1]
        num_macs += Cin * C1[idx] * F * K[idx] * T
        if COUNT_BIAS:
            num_macs += C1[idx] * F * T
        if not CONV_OUTPUT_LENGTH:
            F = F // S[idx]
        num_macs += C1[idx] * F * T * COUNT_BN
    num_macs += C1[-1]**2 * F * T
    print(f"Full-band encoder: {num_macs/1000_000:.1f}M")
    num_macs_total = num_macs

    # Sub-band encoder
    num_macs = 1 * C1[-1] * (4*8 + 7*6 + 11*6 + 20*6 + 40*6) * T
    if COUNT_BIAS:
        num_macs += C1[-1] * 32 * T
    print(f"Sub-band encoder: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Feature merge
    num_macs = (
        C1[-1] * 64 * 32    # Linear
        + C1[-1] * C2 * 32  # Conv
    ) * T
    if COUNT_BIAS:
        num_macs += (C1[-1] * 32 + C2 * 32) * T
    print(f"Feature merge: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # DPRNN
    num_macs = 0
    for _ in range(3):
        num_macs += (
            calc_gru(C2, C2)*2  # Intra RNN
            + C2*2 * C2         # Intra FC
            + C2 * COUNT_LN     # Intra LayerNorm
            + C2 * 1            # Skip connection
            + calc_gru(C2, C2)  # Inter RNN
            + C2**2             # Inter FC
            + C2 * 1            # Skip connection
        ) * 32 * T
        if COUNT_BIAS:
            num_macs += (C2 + C2) * 32 * T
    print(f"DPRNN: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Feature split
    num_macs = (
        C2 * C1[-1] * 32    # Conv
        + C1[-1] * 32 * 64  # Linear
    ) * T
    if COUNT_BIAS:
        num_macs += (C1[-1] * 32 + C1[-1] * 64) * T
    print(f"Feature split: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Sub-band decoder
    num_macs = C1[-1] * (8*2 + 6*3 + 8*5 + 8*10 + 8*20) * T
    if COUNT_BIAS:
        num_macs += (8*2 + 6*3 + 8*5 + 8*10 + 8*20) * T
    print(f"Sub-band decoder: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Full-band decoder
    num_macs = 0
    for idx in range(len(K)-1, -1, -1):
        if not CONVT_INPUT_LENGTH:
            F = F * S[idx] + 1 if idx == 0 else F * S[idx]
        Cout = 2 if idx == 0 else C1[idx-1]
        num_macs += C1[idx] * Cout * F * K[idx] * T
        if COUNT_BIAS:
            num_macs += Cout * F * T
        if CONVT_INPUT_LENGTH:
            F = F * S[idx] + 1 if idx == 0 else F * S[idx]
        if idx > 0:
            num_macs += Cout * F * T * COUNT_BN
    print(f"Full-band decoder: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    # Mask
    num_macs = (
        257 * T * 4     # out_full = spec_noisy * mask_full (complex multiplication)
        + 257 * T * 2   # mask_mag = (mask_sub + mask_full_mag) * 0.5
        + 257 * T * 2   # out_full / mask_full_mag * mask_mag
    )
    print(f"Mask: {num_macs/1000_000:.1f}M")
    num_macs_total += num_macs

    print(f"Total MACs: {num_macs_total/1000_000:.2f}M")


# num_params = (
#     # Full-band encoder
#     2*4*6+4 + 4*16*8+16 + 16*32*6+32
#     + 32*32+32

#     # Sub-band encoder
#     + 32*(4+7+11+20+40)+32*5

#     # Feature merge
#     + 64*32+32
#     + 32*16+16
    
#     # DPE
#     + (
#         (16*16*6+16*6)*2 + 32*16+16 + 16*2
#         + (16*16*6+16*6)*8
#         + (16*16+16)*8
#     ) * 3
    
#     # Feature split
#     + 16*32+32
#     + 32*64+64
    
#     # Full-band decoder
#     + 64*32+32 + 32*16+16 + 8*4+4
#     + 32*16*6+16 + 16*4*8+4 + 4*2*6+2
#     + 64*(2+3+5+10+20)+(2+3+5+10+20)
# )
# print(num_params)