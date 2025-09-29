# LiSenNet
[Paper](https://ieeexplore.ieee.org/document/10446016) [1] | [Github](https://github.com/hyyan2k/LiSenNet/tree/main)  
The official implementation includes input normalization and Griffin-Lim, so it is not streamable. To make the model streamable, input normalization is removed, and instead of Griffin-Lim, the model predicts a complex mask. We configured the training settings identically to FastEnhancer for fair comparison.

[1]: H. Yan, J. Zhang, C. Fan, Y. Zhou, and P. Liu, “LiSenNet: Lightweight sub-band and dual-path modeling for real-time speech enhancement,” in *Proc. IEEE ICASSP*, 2025, pp. 1–5.  

## Training
- Model: LiSenNet
- Dataset: [Voicebank-Demand](../dataset/voicebank-demand.md) at 16kHz sampling rate. 
- Number of GPUs: 1
- Batch size: 64
- Mixed-precision training with fp16: False
- Path to save config, tensorboard logs, and checkpoints: `logs/vbd/16khz/lisennet`
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/lisennet \
  -c configs/others/lisennet.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/lisennet \
  -c configs/others/lisennet.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>

Options:  
- `-n` (Required): Base directory to save configuration, tensorboard logs, and checkpoints.  
- `-c` (Optional): Path to configuration file. If not given, the configuration file in the base directory will be used.  
- `-p` (Optional): Parameters after this will update the configuration.  
- `-f` (Optional): If the base directory already exists and `-c` flag is given, an exception will be raised to avoid overwriting config file. However, enabling this option will force overwriting config file.  

## Resume Training
Suppose you stopped the training.  
To load the saved checkpoint at `logs/vbd/16khz/lisennet` and resume the training, use the code below:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/lisennet</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/lisennet</code></pre>

## Test the training code
Before you start training, we recommend to run the following test code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n delete_it \
  -c configs/others/lisennet.yaml \
  -p train.test=True pesq.interval=1 \
  -f</code></pre>
By setting `train.test=True`, it will execute a training code for only 10 steps. Then it will execute a validation code. Finally, by setting `pesq.interval=1`, it will execute a code for calculating objective metrics and terminate. If the code runs well, you are ready to begin training. You can delete `logs/delete_it` directory after the test.

## About Optimizer
In `configs/others/lisennet.yaml`, you may notice that instead of using `train.optimizer=AdamP` as in FastEnhancer, it uses `train.optimizer=AdamW`. This is because there's no scale-invariant parameter in LiSenNet. It employs LayerNorm and no weight norm is applied. In such case, AdamP is same as AdamW.  

For PReLU weights, one should not apply weight decay, so we set `train.optimizer_groups` appropriately.
