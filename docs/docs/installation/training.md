# (0) Decide Python, CUDA toolkit, and PyTorch versions
Before install, you have to decide which version to install (including Python, CUDA toolkit, and PyTorch).  
Note that `PyTorch>=2.3` is recommended. On `PyTorch<2.3`, `torch.nn.utils.parametrizations.weight_norm` is not implemented, so you have to change the codes and .yaml files.  

First, check CUDA toolkit versions that your nvidia driver supports:
<pre><code><span style="color: #55FF55;font-weight: bold">shahn</span>:<span style="color: #5555FF;font-weight: bold">~</span>$ nvidia-smi | grep "CUDA Version"
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      <span style="color: red;font-weight: bold">CUDA Version</span>: 13.0     |</code></pre>
That is the maximum CUDA toolkit version you can install. In our case, we can choose any version `<= 13.0`.  

Second, visit [here](https://download.pytorch.org/whl/torch/) and decide PyTorch, Python, and CUDA toolkit version.  
Then install Python to your environment.  
For the rest of this document, we will use `torch-2.8.0+cu129-cp313` version, meaning PyTorch `2.8.0`, CUDA toolkit `12.9`, and Python `3.13`.  
You can use your favorite environment manager. We use miniconda as below:
<pre><code>conda create -n fastenhancer python=3.13
conda activate fastenhancer</pre></code>

# (1) Install CUDA toolkit and cuDNN
Download a local runfile of [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) and install.  
In the following example, we will install CUDA toolkit `12.9` in `/home/shahn/.local/cuda-12.9`:
<pre><code>wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run

chmod +x cuda_12.9.1_575.57.08_linux.run

./cuda_12.9.1_575.57.08_linux.run \
  --silent \
  --toolkit \
  --installpath=/home/shahn/.local/cuda-12.9 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page</code></pre></code>
Then, install a tar file of [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for your CUDA version. In the following example, we download cuDNN `8.9.7` for CUDA `12.x` and install as below:
<pre><code>tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz --strip-components=1 -C /home/shahn/.local/cuda-12.9</code></pre>

Finally, set environment variables
<pre><code>export CUDA_HOME=/home/shahn/.local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH</code></pre>
and check:
<pre><code>(fastenhancer) <span style="color: #55FF55;font-weight: bold">shahn</span>:<span style="color: #5555FF;font-weight: bold">~</span>$ which nvcc
/home/shahn/.local/cuda-12.9/bin/nvcc
(fastenhancer) <span style="color: #55FF55;font-weight: bold">shahn</span>:<span style="color: #5555FF;font-weight: bold">~</span>$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:24:57_PDT_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0</code></pre>

# (2) Install PyTorch and Torchaudio
Check which torchaudio version matches your PyTorch version at [here](https://pytorch.org/audio/stable/installation.html#compatibility-matrix).  
Then install approriate version of PyTorch and Torchaudio.  
In the following example, we install PyTorch `2.8.0`, CUDA `12.9` as below:
<pre><code>pip install torch==2.8.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl</code></pre>

# (3) Install other dependencies
<pre><code>pip install jupyter notebook matplotlib tensorboard scipy librosa unidecode einops cython tqdm pyyaml pesq pystoi torch-pesq torchmetrics</code></pre>