# Installation for Training
We tested under:  
- PyTorch==2.7.1, CudaToolkit==11.8, Python==3.13  
- PyTorch==2.7.1, CudaToolkit==12.8, Python==3.13  

Note that we failed under PyTorch==2.8.0.  
- We succeeded to train, calculate metrics, export ONNX spec2spec, and execute ONNXRuntime.  
- However, we failed to export ONNX wav2wav. After downgrading to PyTorch==2.7.1, the problem is solved.  

## (0) Decide Python, CUDA toolkit, and PyTorch versions
Before install, you have to decide which version to install (including Python, CUDA toolkit, and PyTorch).  
Note that `PyTorch>=2.3` is recommended. On `PyTorch<2.3`, `torch.nn.utils.parametrizations.weight_norm` is not implemented, so you have to change the codes and .yaml files. You also have to remove `device_id` argument of `dist.init_process_group` in `train.py`.  

First, check CUDA toolkit versions that your nvidia driver supports:
```
nvidia-smi | grep "CUDA Version"
```
The output should look like this:
```
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
```
That is the maximum CUDA toolkit version you can install. In our case, we can choose any version `<= 13.0`.  

Second, visit [here](https://download.pytorch.org/whl/torch/) and decide PyTorch, Python, and CUDA toolkit version.  
Then install Python to your environment.  
For the rest of this document, we will use `torch-2.7.1+cu128-cp313` version, meaning PyTorch `2.7.1`, CUDA toolkit `12.8`, and Python `3.13`.  
You can use your favorite environment manager. We use miniconda as below:
<pre><code>conda create -n fastenhancer python=3.13 -c conda-forge
conda activate fastenhancer</pre></code>

## (1) Install CUDA toolkit and cuDNN
Download a local runfile of [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) and install.  
In the following example, we will install CUDA toolkit `12.8` in `/home/shahn/.local/cuda-12.8`:
<pre><code>wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run

chmod +x cuda_12.8.1_570.124.06_linux.run

./cuda_12.8.1_570.124.06_linux.run \
  --silent \
  --toolkit \
  --installpath=/home/shahn/.local/cuda-12.8 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page</code></pre></code>
Then, install a tar file of [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for your CUDA version. In the following example, we download cuDNN `8.9.7` for CUDA `12.x` and install as below:
<pre><code>tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz --strip-components=1 -C /home/shahn/.local/cuda-12.8</code></pre>

Finally, set environment variables
<pre><code>export CUDA_HOME=/home/shahn/.local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH</code></pre>
and check:
```
which nvcc

nvcc --version
```
Then the output should look like this:
```
/home/shahn/.local/cuda-12.8/bin/nvcc

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## (2) Install PyTorch and Torchaudio
Check which torchaudio version matches your PyTorch version at [here](https://pytorch.org/audio/stable/installation.html#compatibility-matrix).  
Then install approriate version of PyTorch and Torchaudio.  
In the following example, we install PyTorch `2.7.1`, CUDA `12.8` as below:
```
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl
```

## (3) Install other dependencies
```
pip install jupyter notebook matplotlib tensorboard scipy librosa unidecode einops cython tqdm pyyaml pesq pystoi torch-pesq torchmetrics
```