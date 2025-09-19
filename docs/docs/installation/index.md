# Installation

## Installation for training
Required by `train.py` and `train_torchrun.py`.  

Refer to [Installation for Training](training.md).

## Installation for calculating objective metrics
Required by `metrics_ns.py`.  

First, follow [Installation for Training](training.md).  
Second, install the following pacakges:
<pre><code>pip install torchmetrics jiwer
pip install git+https://github.com/openai/whisper.git</code></pre>

Finally, install [onnxruntime-gpu](https://onnxruntime.ai/docs/install/#python-installs).  
Be careful to install according to your CUDA version.
If GPU version is not available, you can install a CPU version.  
However, DNSMOS and SCOREQ will run on CPU and the `metrics_ns.py` code will run very slow.

## Installation for ONNXRuntime exporting
Required by `export_onnx.py` and `export_onnx_spec.py`.  

First, follow [Installation for Training](training.md).  
Second, install the following pacakges:
<pre><code>pip install onnx onnxsim</code></pre>

## Installation for ONNXRuntime inference
Required by `test_onnx.py` and `test_onnx_spec.py`.  

If you just want to run `test_onnx.py` or `test_onnx_spec.py`, you don't need to install PyTorch or other numerous packages.  
Install dependencies as below:
<pre><code>pip install numpy scipy librosa tqdm</code></pre>

Then install install [onnxruntime](https://onnxruntime.ai/docs/install/#python-installs).  
It doesn't matter whether you intsall a CPU version or a GPU version, because the code will run on CPU anyway.