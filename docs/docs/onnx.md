# ONNX
You can export your own model to ONNX and execute ONNXRuntime.  
You can also download pre-compiled ONNX file and execute ONNXRuntime.  

There are two ways for ONNXRuntime.  

1. Spectrogram-to-spectrogram (spec2spec) version: STFT and iSTFT are done in PyTorch. Only the neural network part is calculated in ONNXRuntime. This version is exported using torch.export.  
2. Waveform-to-waveform (wav2wav) version: STFT and iSTFT are also done in ONNXRuntime. This version is exported using torchdynamo. Only FastEnhancers are successfully exported to this version. For other models, it is either impossible to create a wav2wav version or the ONNXRuntime execution speed is very slow.  

The RTFs in our paper are measured using spec2spec versions.

## Exporting to ONNX and executing ONNXRuntime
Suppose you have trained a model and saved its checkpoints at `logs/fastenhancer_l`.  
The code below exports the model to ONNX, saves to `onnx/fastenhancer_l.onnx`, executes the ONNXRuntime, and calculates the RTF.  
Wav2wav version:
<pre><code>python -m scripts.export_onnx -n fastenhancer_l --onnx-path onnx/fastenhancer_l.onnx</code></pre>

Spec2spec version:
<pre><code>python -m scripts.export_onnx_spec -n fastenhancer_l --onnx-path onnx/fastenhancer_l.onnx</code></pre>

## Executing ONNXRuntime
You can download a pre-compiled ONNX file from [here](https://github.com/aask1357/fastenhancer/releases/).  
If you downloaded a wav2wav version in `onnx/fastenhancer_t.onnx`, run the following code:
<pre><code>python -m scripts.test_onnx --onnx-path onnx/fastenhancer_t.onnx</code></pre>
If you downloaded a spec2spec version in `onnx/fastenhancer_t_spec.onnx`, run the following code:
<pre><code>python -m scripts.test_onnx_spec --onnx-path onnx/fastenhancer_t_spec.onnx</code></pre>

There are some model-specific settings.
- For FastEnhancer-M, you should set `--hop-size 160`:
  <pre><code>python -m scripts.test_onnx --onnx-path onnx/fastenhancer_m.onnx --hop-size 160</code></pre>
- For FastEnhancer-L, you should set `--hop-size 100`:
  <pre><code>python -m scripts.test_onnx --onnx-path onnx/fastenhancer_l.onnx --hop-size 100</code></pre>
- For GTCRN, you should set `--win-type hann-sqrt`:
  <pre><code>python -m scripts.test_onnx_spec --onnx-path onnx/gtcrn_spec.onnx --win-type hann-sqrt</code></pre>
- For other models, use default settings.

## More information about wav2wav version
Let `x` denote a noisy input and `y` denote an enhanced signal.  
Let `n` denote an fft size and `h` denote a hop size.  
In the wav2wav version, at every `i`-th iteration, the model gets `x[i*h+(n-h):i*h+n]` as an input and returns `y[i*h:(i+1)*h]`.  
This means that the input and the output has a delay of `n-h`.  

Why?  

Obviously,  
At the first iteration, the model takes `x[0:n]` and generates an enhanced signal `y[0:n]`.  
At the second iteration, the model takes `x[h:n+h]` and generates an enhanced signal `y[h:n+h]` which is overlap-added to the previous iteration's output.  

However,  
This implies that at the end of first iteration, `y[0:h]` is completed while `y[h:n]` isn't.  
Also, at the beggining of second iteration, only `x[n:n+h]` is a new input. `x[h:n]` was already given at the first iteration.  

So,  
At the first iteration, the model saves `x[h:n]` as its input cache and `y[h:n]` as its output cache.  
At the second iteration, the model gets `x[n:n+h]` as an input. The model concatenate its input cache with the new input to make `x[h:n+h]`. The model generates `y[h:n+h]`. It is overlap-added with the previous output cache `y[h:n]`. Since `y[h:2*h]` is now completed, it is returned. The model caches `x[2*h:n+h]` and `y[2*h:n+h]` for the next iteration.  

The final algorithm is as below:  

- Initially, the model has an input cache `cache_in` whose length is `n-h` and filled with zeros. The model also has an output cache `cache_out` whose length is `n-h` and filled with zeros.  
- At every iterations, the model gets a new input chunk `x` with a length of `h`.  
- The model concatenate the input chunk with its input cache to create an input with a length of `n`:
  <pre><code>x = torch.cat([cache_in, x])</code></pre>
- The model saves the last `n-h` samples as its new input cache:
  <pre><code>cache_in = x[h:n]</code></pre>
- The model generates an enhanced signal `y` with a length of `n`:
  <pre><code>y = model(x)</code></pre>
- The model performs an overlap-add:
  <pre><code>y[0:n-h] += cache_out</code></pre>
- The model saves the last `n-h` samples as its new output cache:
  <pre><code>cache_out = y[h:n]</code></pre>
- The model returns the first `h` samples:
  <pre><code>return y[0:h]</code></pre>
Outside the model, it seems that the model gets an input with a length of `h` and returns an output with a length of `h`. However, you now understand that those input and output are not time-aligned. They have a time difference of `n-h`.