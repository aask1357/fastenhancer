# FastEnhancer
[Paper](https://arxiv.org/abs/2509.21867)

## Training FastEnhancer-Large on Voicebank-Demand 16kHz
- Model: FastEnhancer-Large
- Dataset: [Voicebank-Demand](../dataset/voicebank-demand.md) at 16kHz sampling rate
- Number of GPUs: 4
- Batch size: 16/GPU, total 64
- Mixed-precision training with fp16: True
- Path to save config, tensorboard logs, and checkpoints: `logs/vbd/16khz/fastenhancer_l`
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  -n vbd/16khz/fastenhancer_l \
  -c configs/fastenhancer/l.yaml \
  -p train.batch_size=16 valid.batch_size=16 pesq.batch_size=4 \
  -f</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train_torchrun.py \
  -n vbd/16khz/fastenhancer_l \
  -c configs/fastenhancer/l.yaml \
  -p train.batch_size=16 valid.batch_size=16 pesq.batch_size=4 \
  -f</code></pre>

## Training FastEnhancer-Huge on DNS-Challenge 16kHz
- Model: FastEnhancer-Huge-Noncausal
- Dataset: [DNS-Challenge](../dataset/dns-challenge.md) at 16kHz sampling rate
- Number of GPUs: 4
- Batch size: 16/GPU, total 64
- Mixed-precision training with fp16: True
- Path to save config, tensorboard logs, and checkpoints: `logs/dns/16khz/fastenhancer_h`
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  -n dns/16khz/fastenhancer_h \
  -c configs/fastenhancer/huge_noncausal_dns.yaml \
  -p train.batch_size=16 valid.batch_size=16 pesq.batch_size=4 \
  -f</code></pre>

Options:
- -n (Required): Base directory to save configuration, tensorboard logs, and checkpoints.
- -c (Optional): Path to configuration file. If not given, the configuration file in the base directory will be used.
- -p (Optional): Parameters after this will update the configuration.
- -f (Optional): If the base directory already exists and -c flag is given, an exception will be raised to avoid overwriting config file. However, enabling this option will force overwriting config file.

## Resume Training
Suppose you stopped the training.  
To load the saved checkpoint at `logs/vbd/16khz/fastenhancer_l` and resume the training, use the code below:
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  -n vbd/16khz/fastenhancer_l</code></pre>
or
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train_torchrun.py \
  -n vbd/16khz/fastenhancer_l</code></pre>

## Test the training code
Before you start training, we recommend to run the following test code:
<pre><code>CUDA_VISIBLE_DEVICES=0 python train.py \
  -n delete_it \
  -c configs/fastenhancer/l.yaml \
  -p train.test=True pesq.interval=1 \
  -f</code></pre>
By setting `train.test=True`, it will execute a training code for only 10 steps. Then it will execute a validation code. Finally, by setting `pesq.interval=1`, it will execute a code for calculating objective metrics and terminate. If the code runs well, you are ready to begin training. You can delete `logs/delete_it` directory after the test.

## About optimizer_groups
In `configs/fastenhancer/l.yaml`, you can see a large code block for `train.optimizer_groups`.  
This section is for ones who want to know the details of those lines.  

The lines below set `weight_decay=0` and `projection=disabled` for `weight_g`s of GRUs and `scale` parameter of the final convolution of the decoder. If we apply weight_norm to GRUs, we empirically found that setting `weight_decay=0` to `weight_g`s improves performance. We believe this is because of `tanh` and `sigmoid` functions applied after weight-input multiplication. Also, intuitively, we should not apply `weight_decay` to the final mask prediction.
<pre><code>-
    <span style="color: #569CD6">regex_list</span>:
        - <span style="color: #CE916C">"rf_block\\.\\d\\.rnn\\.parametrizations.+original0$"</span> <span style="color: #6A9955"># GRU weight_g</span>
        - <span style="color: #CE916C">"dec_post\\.3\\.scale"</span>        <span style="color: #6A9955"># scale parameter of the final conv</span>
    <span style="color: #569CD6">weight_decay</span>: <span style="color: #B5CEA8">0</span>
    <span style="color: #569CD6">projection</span>: <span style="color: #CE916C">disabled</span></code></pre>


The lines below set `projection=channelwise` and `projection=layerwise` to appropriate parameters. Note that originally `AdamP(projection=auto)` can automatically handle them by detecting scale-invariant parameters. However, we found that when using mixed-precision training (by setting `train.fp16=True`), `AdamP` often failed to detect scale-invariance because of the numerical error. Therefore, we manually set `projection`s.
<pre><code>-
<span style="color: #569CD6">    regex_list</span>:
        - <span style="color: #CE916C">".+parametrizations.+original1$"</span>  <span style="color: #6A9955"># weight_v</span>
        - <span style="color: #CE916C">"enc_pre\\.0\\.weight"</span>            <span style="color: #6A9955"># conv1d before BN</span>
    <span style="color: #569CD6">projection</span>: <span style="color: #CE916C">channelwise</span>
-
<span style="color: #569CD6">   regex_list</span>:
        - <span style="color: #CE916C">"dec_post\\.3\\.weight"</span>           <span style="color: #6A9955"># final conv</span>
    <span style="color: #569CD6">projection</span>: <span style="color: #CE916C">layerwise</span></code></pre>
