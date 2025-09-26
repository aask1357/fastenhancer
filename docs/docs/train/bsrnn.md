# BSRNN
- Original paper that proposed BSRNN: [paper](https://arxiv.org/abs/2209.15174)[^1]  
- First paper that used BSRNN to noise suppression: [paper](https://arxiv.org/abs/2212.00406)[^2]  
- BSRNN model size scaling: [paper](https://arxiv.org/abs/2406.04269)[^3] | [github](https://github.com/Emrys365/se-scaling)  

We implemented a streaming BSRNN with batch normalization. We follow [2] for the configurations of different sizes.  

[^1]: Y. Luo and J. Yu, “Music source separation with band-split RNN”, *IEEE/ACM Trans. ASLP*, vol. 31, pp. 1893-1901, 2023.  
[^2]: J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech enhancement with band-split RNN,” in *Proc. Interspeech*, 2023, pp. 2483–2487.  
[^3]: W. Zhang, K. Saijo, J.-w. Jung, C. Li, S. Watanabe, and Y. Qian, “Beyond performance plateaus: A comprehensive study on scalability in speech enhancement,” in *Proc. Interspeech*, 2024, pp. 1740-1744.  

## Training
- Model: BSRNN-xxt
- Dataset: [Voicebank-Demand](voicebank-demand.md) at 16kHz sampling rate. 
- Number of GPUs: 1
- Batch size: 64
- Mixed-precision training with fp16: False
- Path to save config, tensorboard logs, and checkpoints: `logs/vbd/16khz/bsrnn_xxt`
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/bsrnn_xxt \
  -c configs/others/bsrnn_xxt.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/bsrnn_xxt \
  -c configs/others/bsrnn_xxt.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>

Options:
- -n (Required): Base directory to save configuration, tensorboard logs, and checkpoints.
- -c (Optional): Path to configuration file. If not given, the configuration file in the base directory will be used.
- -p (Optional): Parameters after this will update the configuration.
- -f (Optional): If the base directory already exists and -c flag is given, an exception will be raised to avoid overwriting config file. However, enabling this option will force overwriting config file.

## Resume Training
Suppose you stopped the training.  
To load the saved checkpoint at `logs/vbd/16khz/bsrnn_xxt` and resume the training, use the code below:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/bsrnn_xxt</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/bsrnn_xxt</code></pre>

## Test the training code
Before you start training, we recommend to run the following test code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n delete_it \
  -c configs/others/bsrnn_xxt.yaml \
  -p train.test=True pesq.interval=1 \
  -f</code></pre>
By setting `train.test=True`, it will execute a training code for only 10 steps. Then it will execute a validation code. Finally, by setting `pesq.interval=1`, it will execute a code for calculating objective metrics and terminate. If the code runs well, you are ready to begin training. You can delete `logs/delete_it` directory after the test.

## About optimizer_groups
In `configs/others/bsrnn_xxt.yaml`, you may notice that instead of using `train.optimizer=AdamP` as in FastEnhancer, it uses `train.optimizer=AdamW`. This is because there's no scale-invariant parameter in BSRNN. It employs pre-activation and no weight norm is applied. In such case, AdamP is same as `AdamW.  
