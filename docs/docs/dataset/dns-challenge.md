# DNS-Challenge Datasets
DNS-Challenge datasets are noise suppression dataset.  

- DNS-Challenge 1 (Interspeech 2020): Widebank(16kHz), 500 hours of clean speech from LibriVox audiobooks. It contains a synthetic dev test set with `clean` and `noisy` pairs that can be used for calculating intrusive objective metrics. It also contains a real recording dev test set and a blind test set that can be used for calculating non-intrusive objective metrics and subjective metrics.  
- DNS-Challenge 2 (ICASSP 2021): Wideband(16kHz), 760 hours of clean speech (singing voice, emotion data, and non-English are added).  
- DNS-Challenge 3 (Interspeech 2021): Wideband(16kHz) & Fullband(48kHz).  
- DNS-Challenge 4 (ICASSP 2022): Fullband(48kHz). Personalized train dataset added.  
- DNS-Challenge 5 (ICASSP 2023): Fullband(48kHz). Headset dataset added.  

For 16kHz dataset, we recommend to download DNS-Challenge 3 wideband train dataset and DNS-Challenge 1 synthetic testset.  
For 48kHz dataset, we recommend to download DNS-Challenge 5 speakerphone train dataset. Note that no test set with `clean` and `noisy` pairs is provided. Test sets with only `noisy` files are provided.  

## Preparing dataset
### Download
Download the dataset from [here](https://github.com/microsoft/DNS-Challenge/).  
For DNS-Challenge 1 synthetic dev test set, we provide a pre-processed version [here](https://github.com/aask1357/fastenhancer/releases/tag/test-data-v1)

### Downsample
If needed, downsample the dataset using `scripts/resample.py`.  
For example, if you want to downsample to 24kHz, run the code below:
<pre><code>python -m scripts.resample --to-sr 24000 --from-dir ~/Datasets/DNS_Challenge/dataset_fullband --to-dir ~/Datasets/DNS_Challenge/dataset_24khz</code></pre>

## Modify Configuration file
You have to change the dataset path and sampling rate of configuration file.
Since there's no official testset composed of `clean` and `noisy` pairs for fullband, you can train with DNS-Challenge dataset and validate with other dataset.
For example, in `configs/fastenhancer/huge_dns.yaml`, we trained `FastEnhancer-Huge-noncausal` at 24kHz using DNS-Challenge dataset and validated using Voicebank-demand testset.  
Modify various configurations in `data` section of the config file as you wish.  
If the dataset loading is too slow, you may consider increasing `train.num_workers`. If the speed is still slow, we recommend to write a code to synthesize dataset in advance and train using the synthesized dataset.  

## Dataset Code
For DNS-Challenge dataset, we load clean speech, noise, and optionally RIR. They are later mixed on-the-fly at the training code.  
To check or modify the dataset code, see `utils/data/ns_on_the_fly.py`.

## Training Code
To check or modify the training code for DNS-Challenge, see `wrappers/ns_on_the_fly.py`.
Clean speech and noise pairs are loaded and mixed on-the-fly to generate noisy.  

If you want to add RIR to clean speech, you have to modify the code.  

- `config.yaml`  
  - Set `data.reverb_prob` higher than 0 and leq than 1.  
  - Set `data.rir_length` to the max length of rir. For DNS-Challenge synthetic RIR, set it to 2 seconds (32000 for 16kHz and 96000 for 48khz).  
- `wrappers/ns_on_the_fly.py`  
  - Add `rir` to `self.keys`  
  - Load `batch['rir']`, and give it to `self.snr_mixer`.  

In this case, the rir-convolved clean speech becomes the target, which means your model doen't perform dereverberation.  
If you want to do dereverberation along with noise suppression, you have to implement on your own.  
Some papers preserve only the first 100ms reflections and use that as a target (GTCRN, UL-UNAS).  
In URGENT challenges, they find the rir_start_index and preserve the 50ms reflections from that starting point.