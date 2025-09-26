# Objective Metrics
We support
- DNSMOS P.808 and P.835 (SIG, BAK, OVL)
- SCOREQ ([github](https://github.com/alessandroragano/scoreq))
  - It provides two domains: Natural Speech domain trained for real recordings, and Synthetic Speech domain trained for TTS.
  - We use Natural Speech domain.
  - It provides two modes: no-reference (NR) and non-matching reference (NMR). NR mode is a non-intrusive mode. NMR mode takes reference and degraded signals. However, the reference doesn't have to be the clean speech pair of the degraded signal.
  - We use full reference mode, meaning clean speech is given as the reference of the NMR mode.
- SISDR
  - On some implementations, mean values of reference and degraded signal are substracted. However, we don't perform mean substraction.
- PESQ ([github](https://github.com/ludlows/PESQ))
  - We use P.862.2 without Corrigendum 2. For more information, see the following [paper](https://arxiv.org/abs/2505.19760) and [github](https://github.com/audiolabs/PESQ).
- STOI and ESTOI ([github](https://github.com/mpariente/pystoi))
- WER
  - [Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) is used.
  - For Voicebank-Demand testset, there exists transcript, so we can calculate WER.
  - For DNS-Challenge dev testset (Interspeech2020, synthetic), transcript is not provided. Furthermore, each clean speech is a random crop of a long audio, so the beginning and end of the speech are cut off. Therefore, it is impossible to create a transcript. We don't calculate WER for DNS-Challenge.

## Calculating objective metrics
Assume you want to calculate objective metrics for a model in `logs/fastenhancer_l`.  
For Voicebank-Demand, run the following code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python -m scripts.metrics_ns -n fastenhancer_l --transcript-dir PATH-TO-TRANSCRIPT</code></pre>
For DNS-Challenge, run the following code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python -m scripts.metrics_ns -n fastenhancer_l --wer False</code></pre>
If you want to use cuda:1, don't set `-d cuda:1`. Set `CUDA_VISIBLE_DEVICES=1` and `-d cuda:0`.