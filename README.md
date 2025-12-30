# Introduction
Official repository of "FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement."  
[Paper](https://arxiv.org/abs/2509.21867) | [Documentation](https://aask1357.github.io/fastenhancer/)

# Install
Please refer to [document](https://aask1357.github.io/fastenhancer/installation).

# Datasets
Please refer to [document](https://aask1357.github.io/fastenhancer/dataset).

# Training
Please refer to [document](https://aask1357.github.io/fastenhancer/train).

# Inference
## PyTorch Inference
Pytorch checkpoints and tensorboard logs are provided in [releases](https://github.com/aask1357/fastenhancer/releases).  
Please refer to [document](https://aask1357.github.io/fastenhancer/metrics) for calculating objective metrics.  
Please refer to [document](https://aask1357.github.io/fastenhancer/pytorch) for pytorch inference.

## ONNXRuntime Inference
ONNX models are provided in [releases](https://github.com/aask1357/fastenhancer/releases).  
Please refer to [document](https://aask1357.github.io/fastenhancer/onnx) for streaming inference using ONNXRuntime. 

# Results
## Voicebank-Demand 16kHz
* Except for GTCRN, we trained each model five times with five different seed and report the average scores.
<p align="center"><b>Table 1.</b> Performance on Voicebank-Demand testset.</p>
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Para.<br>(K)</th>
      <th rowspan="2">MACs</th>
      <th rowspan="2">RTF<br>(Xeon)</th>
      <th rowspan="2">RTF<br>(M1)</th>
      <th rowspan="2">DNSMOS<br>(P.808)</th>
      <th colspan="3">DNSMOS (P.835)</th>
      <th rowspan="2">SCOREQ</th>
      <th rowspan="2">SISDR</th>
      <th rowspan="2">PESQ</th>
      <th rowspan="2">STOI</th>
      <th rowspan="2">ESTOI</th>
      <th rowspan="2">WER</th>
    </tr>
    <tr>
      <th>SIG</th>
      <th>BAK</th>
      <th>OVL</th>
    </tr>
  </thead>
  <tbody align=center>
    <tr>
      <td>GTCRN<sup>a</sup></td>
      <td><strong>24</strong></td>
      <td><strong>40M</strong></td>
      <td>0.060</td>
      <td>0.042</td>
      <td>3.43</td>
      <td>3.36</td>
      <td>4.02</td>
      <td>3.08</td>
      <td>0.330</td>
      <td>18.8</td>
      <td>2.87</td>
      <td>0.940</td>
      <td>0.848</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>LiSenNet<sup>b</sup></td>
      <td>37</td>
      <td>56M</td>
      <td>-</td>
      <td>-</td>
      <td>3.34</td>
      <td>3.30</td>
      <td>3.90</td>
      <td>2.98</td>
      <td>0.425</td>
      <td>13.5</td>
      <td>3.08</td>
      <td>0.938</td>
      <td>0.842</td>
      <td>3.7</td>
    </tr>
    <tr>
      <td>LiSenNet<sup>c</sup></td>
      <td>37</td>
      <td>56M</td>
      <td>0.034</td>
      <td>0.028</td>
      <td>3.42</td>
      <td>3.34</td>
      <td><strong>4.03</strong></td>
      <td>3.07</td>
      <td>0.335</td>
      <td>18.5</td>
      <td>2.98</td>
      <td>0.941</td>
      <td>0.851</td>
      <td>3.4</td>
    </tr>
    <tr>
      <td>FSPEN<sup>d</sup></td>
      <td>79</td>
      <td>64M</td>
      <td>0.046</td>
      <td>0.038</td>
      <td>3.40</td>
      <td>3.33</td>
      <td>4.00</td>
      <td>3.05</td>
      <td>0.324</td>
      <td>18.4</td>
      <td>3.00</td>
      <td>0.942</td>
      <td>0.850</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>BSRNN<sup>d</sup></td>
      <td>334</td>
      <td>245M</td>
      <td>0.059</td>
      <td>0.062</td>
      <td>3.44</td>
      <td>3.36</td>
      <td>4.00</td>
      <td>3.07</td>
      <td>0.303</td>
      <td>18.9</td>
      <td>3.06</td>
      <td>0.942</td>
      <td>0.855</td>
      <td>3.4</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_B</td>
      <td>92</td>
      <td>262M</td>
      <td><strong>0.022</strong></td>
      <td><strong>0.026</strong></td>
      <td><strong>3.47</strong></td>
      <td><strong>3.38</strong></td>
      <td>4.02</td>
      <td><strong>3.10</strong></td>
      <td><strong>0.285</strong></td>
      <td><strong>19.0</strong></td>
      <td><strong>3.13</strong></td>
      <td><strong>0.945</strong></td>
      <td><strong>0.861</strong></td>
      <td><strong>3.2</strong></td>
    </tr>
    <tr><td colspan=15></td></tr>
    <tr>
      <td><i>FastEnhancer</i>_T</td>
      <td><strong>22</strong></td>
      <td><strong>55M</strong></td>
      <td><strong>0.012</strong></td>
      <td><strong>0.013</strong></td>
      <td>3.42</td>
      <td>3.34</td>
      <td>4.01</td>
      <td>3.06</td>
      <td>0.334</td>
      <td>18.6</td>
      <td>2.99</td>
      <td>0.940</td>
      <td>0.850</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_B</td>
      <td>92</td>
      <td>262M</td>
      <td>0.022</td>
      <td>0.026</td>
      <td>3.47</td>
      <td>3.38</td>
      <td>4.02</td>
      <td>3.10</td>
      <td>0.285</td>
      <td>19.0</td>
      <td>3.13</td>
      <td>0.945</td>
      <td>0.861</td>
      <td>3.2</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_S</td>
      <td>195</td>
      <td>664M</td>
      <td>0.034</td>
      <td>0.048</td>
      <td>3.49</td>
      <td>3.40</td>
      <td>4.03</td>
      <td>3.12</td>
      <td>0.265</td>
      <td>19.2</td>
      <td>3.19</td>
      <td>0.947</td>
      <td>0.866</td>
      <td>3.2</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_M</td>
      <td>492</td>
      <td>2.9G</td>
      <td>0.101</td>
      <td>0.173</td>
      <td>3.48</td>
      <td>3.39</td>
      <td>4.02</td>
      <td>3.11</td>
      <td>0.243</td>
      <td>19.4</td>
      <td>3.24</td>
      <td>0.950</td>
      <td>0.873</td>
      <td><strong>2.8</strong>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_L</td>
      <td>1105</td>
      <td>11G</td>
      <td>0.313</td>
      <td>0.632</td>
      <td><strong>3.53</strong></td>
      <td><strong>3.44</strong></td>
      <td><strong>4.04</strong></td>
      <td><strong>3.16</strong></td>
      <td><strong>0.239</strong></td>
      <td><strong>19.6</strong></td>
      <td><strong>3.26</strong></td>
      <td><strong>0.952</strong></td>
      <td><strong>0.877</strong></td>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
<p><sup>a</sup> Evaluated using the official checkpoint.<br>
<sup>b</sup> Trained using the official training code. Not streamable because of input normalization and griffin-lim. Thus, RTFs are not reported.<br>
<sup>c</sup> To make the model streamable, input normalization and griffin-lim are removed. Trained following the experimental setup of FastEnhancer (same loss function, same optimizer, etc. Only differences are the model architectures).<br>
<sup>d</sup> Re-implemented and trained following the experimental setup of FastEnhancer (same loss function, same optimizer, etc. Only differences are the model architectures).</p>

## DNS-Challenge 16kHz
* Trained using DNS-Challenge-3 wideband training dataset.
  * Without `emotional_speech` and `singing_voice`.
  * With VCTK-0.92 clean speech except `p232` and `p257` speakers.
  * RIRs were not convolved to the clean speech.
  * Unlike in Voicebank-Demand, we didn't use PESQLoss.
* Tested using DNS-Challenge-1 dev-testset-synthetic-no-reverb dataset.
* We trained each model only once with one random seed.  

<p align="center"><b>Table 2.</b> Performance on DNS-Challenge1 dev-testset-synthetic-no-reverb.</p>
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Para.<br>(K)</th>
      <th rowspan="2">MACs</th>
      <th rowspan="2">RTF<br>(Xeon)</th>
      <th rowspan="2">RTF<br>(M1)</th>
      <th rowspan="2">DNSMOS<br>(P.808)</th>
      <th colspan="3">DNSMOS (P.835)</th>
      <th rowspan="2">SCOREQ</th>
      <th rowspan="2">SISDR</th>
      <th rowspan="2">PESQ</th>
      <th rowspan="2">STOI</th>
      <th rowspan="2">ESTOI</th>
    </tr>
    <tr>
      <th>SIG</th>
      <th>BAK</th>
      <th>OVL</th>
    </tr>
  </thead>
  <tbody align=center>
    <tr>
      <td>GTCRN<sup>a</sup></td>
      <td><strong>24</strong></td>
      <td><strong>40M</strong></td>
      <td>0.060</td>
      <td>0.042</td>
      <td>3.85</td>
      <td>3.35</td>
      <td>3.98</td>
      <td>3.05</td>
      <td>0.551</td>
      <td>14.8</td>
      <td>2.26</td>
      <td>0.934</td>
      <td>0.871</td>
    </tr>
    <tr>
      <td>LiSenNet<sup>b</sup></td>
      <td>37</td>
      <td>56M</td>
      <td>0.034</td>
      <td>0.028</td>
      <td>3.82</td>
      <td>3.39</td>
      <td>4.08</td>
      <td>3.14</td>
      <td>0.487</td>
      <td>16.3</td>
      <td>2.58</td>
      <td>0.947</td>
      <td>0.893</td>
    </tr>
    <tr>
      <td>FSPEN<sup>b</sup></td>
      <td>79</td>
      <td>64M</td>
      <td>0.046</td>
      <td>0.038</td>
      <td>3.82</td>
      <td>3.37</td>
      <td>4.09</td>
      <td>3.13</td>
      <td>0.510</td>
      <td>15.8</td>
      <td>2.43</td>
      <td>0.943</td>
      <td>0.885</td>
    </tr>
    <tr>
      <td>BSRNN<sup>b</sup></td>
      <td>334</td>
      <td>245M</td>
      <td>0.059</td>
      <td>0.062</td>
      <td>3.89</td>
      <td>3.41</td>
      <td>4.11</td>
      <td>3.18</td>
      <td>0.441</td>
      <td><strong>16.7</strong></td>
      <td>2.61</td>
      <td>0.951</td>
      <td>0.901</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_B</td>
      <td>92</td>
      <td>262M</td>
      <td><strong>0.022</strong></td>
      <td><strong>0.026</strong></td>
      <td><strong>3.92</strong></td>
      <td><strong>3.43</strong></td>
      <td><strong>4.12</strong></td>
      <td><strong>3.20</strong></td>
      <td><strong>0.396</strong></td>
      <td><strong>16.7</strong></td>
      <td><strong>2.69</strong></td>
      <td><strong>0.953</strong></td>
      <td><strong>0.903</strong></td>
    </tr>
    <tr><td colspan=14></td></tr>
    <tr>
      <td><i>FastEnhancer</i>_T</td>
      <td><strong>22</strong></td>
      <td><strong>55M</strong></td>
      <td><strong>0.012</strong></td>
      <td><strong>0.013</strong></td>
      <td>3.81</td>
      <td>3.35</td>
      <td>4.07</td>
      <td>3.10</td>
      <td>0.522</td>
      <td>15.4</td>
      <td>2.43</td>
      <td>0.940</td>
      <td>0.879</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_B</td>
      <td>92</td>
      <td>262M</td>
      <td>0.022</td>
      <td>0.026</td>
      <td>3.92</td>
      <td>3.43</td>
      <td>4.12</td>
      <td>3.20</td>
      <td>0.396</td>
      <td>16.7</td>
      <td>2.69</td>
      <td>0.953</td>
      <td>0.903</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_S</td>
      <td>195</td>
      <td>664M</td>
      <td>0.034</td>
      <td>0.048</td>
      <td>3.96</td>
      <td>3.46</td>
      <td>4.13</td>
      <td>3.23</td>
      <td>0.373</td>
      <td>17.5</td>
      <td>2.79</td>
      <td>0.960</td>
      <td>0.914</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_M</td>
      <td>492</td>
      <td>2.9G</td>
      <td>0.101</td>
      <td>0.173</td>
      <td>3.98</td>
      <td>3.48</td>
      <td>4.14</td>
      <td>3.26</td>
      <td>0.345</td>
      <td>18.4</td>
      <td>2.78</td>
      <td>0.965</td>
      <td>0.924</td>
    </tr>
    <tr>
      <td><i>FastEnhancer</i>_L</td>
      <td>1105</td>
      <td>11G</td>
      <td>0.313</td>
      <td>0.632</td>
      <td><strong>4.02</strong></td>
      <td><strong>3.51</strong></td>
      <td><strong>4.16</strong></td>
      <td><strong>3.29</strong></td>
      <td><strong>0.298</strong></td>
      <td><strong>19.5</strong></td>
      <td><strong>2.94</strong></td>
      <td><strong>0.971</strong></td>
      <td><strong>0.935</strong></td>
    </tr>
  </tbody>
</table>
<p><sup>a</sup> Evaluated using the official checkpoint. It should be noted that this model was trained for both noise suppression and de-reverberation, whereas FastEnhancers were trained only for noise suppression. If GTCRN is trained for noise suppression only, its performance may be higher.<br>
<sup>b</sup> Re-implemented and trained following the experimental setup of FastEnhancer (same loss function, same optimizer, etc. Only differences are the model architectures).</p>
