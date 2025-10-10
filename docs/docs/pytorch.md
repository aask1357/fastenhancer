# PyTorch Inference
In this page, we provide an example script that enhances all audio files in a given directory using a pre-trained model and saves the output to the given directory.  
Lets assume that  
- A model checkpoint is in `logs/fastenhancer_l`.  
- Input noisy audio files are in '~/Datasets/noisy'.  
- You want to save the output enhanced audio files in 'enhanced'.

Then, run the following code:  
<pre><code>CUDA_VISIBLE_DEVIECS=0 python -m scripts.test_pytorch -n fastenhancer_l -i ~/Datasets/noisy -o enhanced</code></pre>