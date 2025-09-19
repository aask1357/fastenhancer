# Voicebank-Demand Dataset
Voicebank-Demand, also known as VCTK-Demand, is a noise suppression dataset with a sampling rate of 48kHz.  
There are two train datasets: one is a 28-speaker version, and the other is a 56-speaker version.  
In many papers, including ours, the 28-speaker version is used.  

## Preparing dataset
### Download
Download the train data, test data, and logfiles from [here](https://datashare.ed.ac.uk/handle/10283/2791).  
Download a trainscript file of the testset from [here](https://github.com/aask1357/fastenhancer/releases/download/v1.0.0/transcript_testset.txt).

### Resample
If needed, downsample the dataset using `resample.py`.  
For example, if you want to downsample to 16kHz, run the code below:
<pre><code>python -m resample --to-sr 16000 --from-dir ~/Datasets/voicebank-demand/48k --to-dir ~/Datasets/voicebank-demand/16k</code></pre>
After downloading and resampling, the directory may look like this:
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