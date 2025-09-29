# FSPEN
[Paper](https://ieeexplore.ieee.org/document/10446016)[^1]  
Since there's no official implementation, we faithfully re-implemented the model architecture following the paper and configured the training settings identically to FastEnhancer for fair comparison.

[^1]: L. Yang, W. Liu, R. Meng, G. Lee, S. Baek, and H.-G. Moon, “Fspen: an ultra-lightweight network for real time speech enahncment,” in *Proc. IEEE ICASSP*, 2024, pp. 10671–10675.  

## Training
- Model: FSPEN
- Dataset: [Voicebank-Demand](../dataset/voicebank-demand.md) at 16kHz sampling rate. 
- Number of GPUs: 1
- Batch size: 64
- Mixed-precision training with fp16: False
- Path to save config, tensorboard logs, and checkpoints: `logs/vbd/16khz/fspen`
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/fspen \
  -c configs/others/fspen.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/fspen \
  -c configs/others/fspen.yaml \
  -p train.batch_size=64 valid.batch_size=64 \
  -f</code></pre>

Options:
- -n (Required): Base directory to save configuration, tensorboard logs, and checkpoints.
- -c (Optional): Path to configuration file. If not given, the configuration file in the base directory will be used.
- -p (Optional): Parameters after this will update the configuration.
- -f (Optional): If the base directory already exists and -c flag is given, an exception will be raised to avoid overwriting config file. However, enabling this option will force overwriting config file.

## Resume Training
Suppose you stopped the training.  
To load the saved checkpoint at `logs/vbd/16khz/fspen` and resume the training, use the code below:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n vbd/16khz/fspen</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train_torchrun.py \
  -n vbd/16khz/fspen</code></pre>

## Test the training code
Before you start training, we recommend to run the following test code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n delete_it \
  -c configs/others/fspen.yaml \
  -p train.test=True pesq.interval=1 \
  -f</code></pre>
By setting `train.test=True`, it will execute a training code for only 10 steps. Then it will execute a validation code. Finally, by setting `pesq.interval=1`, it will execute a code for calculating objective metrics and terminate. If the code runs well, you are ready to begin training. You can delete `logs/delete_it` directory after the test.

## About optimizer_groups
In `configs/fastenhancer/l.yaml`, take a look at `train.optimizer_groups`. There are only two scale-invariant parameters in FSPEN, and we manually set them.
