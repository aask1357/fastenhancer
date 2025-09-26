# Training
Before you start training, make sure to [prepare datasets](../dataset/index.md).  

## Training code
Codes for training is in `wrappers/ns.py` (for Voicebank-Demand) and `wrappers/ns_on_the_fly.py` (for DNS-Challenge).  
Training codes perform the following jobs.  
1. For every epoch, they train a model and writes losses to the tensorboard. If `train.plot_params_and_grad` is set to true, they write parameters and gradients to the tensorboard.
2. For every epoch, they validates and writes losses to the tensorboard.
3. For every `train.save_interval` epochs, it saves the checkpoints.
4. For every `infer.interval` epochs, it inferences some samples and writes audios to the tensorboard.
5. For every `pesq.interval` epochs, it calculates objective metrics (PESQ and STOI) and writes to the tensorboard.

You can set those `interval`s in `config/*.yaml` files.

## Training recipes
- [FastEnhancer](fastenhancer.md)
- [BSRNN](bsrnn.md)
- [FSPEN](fspen.md)
- [LiSenNet](lisennet.md)

## Experience sharing for training
Except for Voicebank-Demand at 16kHz sampling rate, we recommend not to use PESQLoss. The reasons are:  
1. It harms stable training.  
2. It doesn't improve other metrics so much (in VoiceBank-Demand @ 16kHz, other metrics marginally improves, so we included it in our paper).
3. The loss includes multiple IIR filter calculations, resulting in increased training time.  

In our experiments, we found that using MetricGAN instead of PESQLoss shows inferior results. MetricGAN achieved a smaller PESQ improvement than PESQLoss and degraded other objective metrics. However, these results may vary depending on the loss functions, batch size, datasets, and models.