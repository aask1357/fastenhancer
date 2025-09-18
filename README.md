# Environment
We tested under:
- CUDA=11.8, torch=2.7, python=3.13
- CUDA=12.9, torch=2.8, python=3.13
It may work in other environments, but not guaranteed.

# Install
## Install for training
First, install CUDA toolkit and cuDNN.
Second, install [PyTorch](https://pytorch.org/get-started/locally/) along with torchaudio.
Third, install dependencies as below:
<pre><code>pip install jupyter notebook matplotlib tensorboard scipy librosa unidecode einops cython tqdm pyyaml onnx onnxsim pesq pystoi torch-pesq torchmetrics
</code></pre>
Fourth, install [onnxruntime-gpu](https://onnxruntime.ai/docs/install/#python-installs).
## Install for ONNXRuntime inference
Install dependencies as below:
<pre><code>pip install matplotlib scipy librosa tqdm pyyaml onnx onnxsim 
</code></pre>
Then install install [onnxruntime](https://onnxruntime.ai/docs/install/#python-installs).
You may install GPU version. The inference code (test_onnx.py) will run on a single CPU thread.