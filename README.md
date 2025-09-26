#SINQ: Sinkhorn Normalized Quantization for LLMs

## Welcome 
Welcome to the Sinkhorn Normalized Quantization (SINQ) repository. Here you'll find code to reproduce results our arxiv paper (link to be added soon). 

## Intro
Example Usage:

```bash
    pip install -r req.txt
    pip install -e .
    cd tests
    python ./quant_model_eval.py
```

This will use SINQ to quantize Qwen3-1.7B to 4bits with group size 64, 1D tiling and dual-scale + shift parameterization.

For the uniform, uncalibrated results simply run 
```bash
    python ./quant_model_eval.py --model_name Qwen/Qwen3-1.7B
```

For non-uniform use e.g.
```bash
    python ./quant_model_eval.py --method sinq_nf4 --model_name ...
```

For calibrated use e.g.
```bash
    python ./quant_model_eval.py --method sinq_awq_l1_quantAux --model_name ...
```

Additional flags are '--tiling_mode' that can be '1D' or '2D', '--nbits', '--group_size'.

## Related Repos
This code builds on https://github.com/Efficient-ML/Qwen3-Quantization
and https://github.com/mobiusml/hqq 
Find their original licenses in the corresponding folders.


