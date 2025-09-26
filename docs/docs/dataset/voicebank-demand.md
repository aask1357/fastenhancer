# Voicebank-Demand Dataset
Voicebank-Demand, also known as VCTK-Demand, is a noise suppression dataset with a sampling rate of 48kHz.  
There are two train datasets: one is a 28-speaker version, and the other is a 56-speaker version.  
In many papers, including ours, the 28-speaker version is used.  

## Preparing dataset
### Download
Download the train data, test data, and logfiles from [here](https://datashare.ed.ac.uk/handle/10283/2791).  
Download a trainscript file of the testset from [here](https://github.com/aask1357/fastenhancer/releases/tag/test-data-v1).

### Downsample
If needed, downsample the dataset using `scripts/resample.py`.  
For example, if you want to downsample to 16kHz, run the code below:
<pre><code>python -m scripts.resample --to-sr 16000 --from-dir ~/Datasets/voicebank-demand/48k --to-dir ~/Datasets/voicebank-demand/16k</code></pre>
After downloading, the directory may look like this:
<pre><code>voicebank-demand
├─ 16k
|  ├─ clean_testset_wav
|  ├─ clean_trainset_28spk_wav
|  ├─ noisy_testset_wav
|  └─ noisy_trainset_28spk_wav
├─ 48k
|  ├─ clean_testset_wav
|  ├─ clean_trainset_28spk_wav
|  ├─ noisy_testset_wav
|  └─ noisy_trainset_28spk_wav
└─ logfiles
   ├─ log_readme.txt
   ├─ log_testset.txt
   ├─ log_trainset_28spk.txt
   └─ transcript_testset.txt</code></pre>

## Modify Configuration file
You have to change the dataset path and sampling rate of configuration file.
For example, to train FastEnhancer-B, change `data` section in `configs/fastenhancer/b.yaml`.

## Dataset Code
For Voicebank-demand dataset, we load clean and noisy speech pairs.  
To check or modify the dataset code, see `utils/data/voicebank_demand.py`.

## Training Code
To check or modify the training code for Voicebank-Demand, see `wrappers/ns.py`.  
Clean and noisy pairs are loaded for training.