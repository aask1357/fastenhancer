# Installation

## Installation for all features
First, follow [Installation for Training](training.md).  
Second, install the following packages:
<pre><code>pip install torchmetrics jiwer onnx onnxsim onnxscript
pip install git+https://github.com/openai/whisper.git</code></pre>
Third, install [onnxruntime-gpu](https://onnxruntime.ai/docs/install/#python-installs).  
Make sure to install the version that matches your CUDA version.  
If you cannot install the GPU version, you can install the CPU version instead.  
However, DNSMOS and SCOREQ will run on CPU and the `metrics_ns.py` code will run very slow.

## Minimal installation for training
Required by `train.py` and `train_torchrun.py`.  

Refer to [Installation for Training](training.md).

## Minimal installation for calculating objective metrics
Required by `metrics_ns.py`.  

First, follow [Installation for Training](training.md).  
Second, install the following pacakges:
<pre><code>pip install torchmetrics jiwer
pip install git+https://github.com/openai/whisper.git</code></pre>
Finally, install [onnxruntime-gpu](https://onnxruntime.ai/docs/install/#python-installs).  
Make sure to install the version that matches your CUDA version.  
If you cannot install the GPU version, you can install the CPU version instead.  
However, DNSMOS and SCOREQ will run on CPU and the `metrics_ns.py` code will run very slow.

## Minimal installation for ONNX exporting
Required by `scripts/export_onnx.py` and `scripts/export_onnx_spec.py`.  

First, follow [Installation for Training](training.md).  
Second, install the following pacakges:
<pre><code>pip install onnx onnxsim onnxscript</code></pre>

## Minimal Installation for ONNXRuntime (spec2spec version)
Required by `scripts/test_onnx_spec.py`.  

First, install [PyTorch](https://pytorch.org/get-started/locally/). It doesn't need to be GPU version.  
Second, install the following pacakges:
<pre><code>pip install numpy scipy librosa tqdm</code></pre>
Finally, install install [onnxruntime](https://onnxruntime.ai/docs/install/#python-installs).  
It doesn't matter whether you intsall a CPU version or a GPU version. Even if you install a GPU version, the code will run on CPU anyway.

## Minimal Installation for ONNXRuntime (wav2wav version)
Required by `scripts/test_onnx.py`.  

You don't need to install PyTorch in this case.
First, install the following pacakges:
<pre><code>pip install numpy scipy librosa tqdm</code></pre>
Then install install [onnxruntime](https://onnxruntime.ai/docs/install/#python-installs).  
It doesn't matter whether you intsall a CPU version or a GPU version. Even if you install a GPU version, the code will run on CPU anyway.