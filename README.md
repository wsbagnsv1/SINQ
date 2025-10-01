[![arXiv](https://img.shields.io/badge/arXiv-2509.22944-b31b1b.svg)](https://arxiv.org/abs/2509.22944)

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img src="imgs/logo.png" alt="SINQ Logo" width="110"></td>
    <td style="vertical-align: middle;"><h1>SINQ: Sinkhorn-Normalized Quantization for LLMs</h1></td>
  </tr>
</table>



> ⚡️ **A fast, plug-and-play, model-agnostic quantization technique** delivering **state-of-the-art performance** for Large Language Models **without sacrificing accuracy.**

> 💡 **Want to run a large model on your GPU but don’t have enough memory?** With **SINQ**, you can deploy models that would otherwise be too big **drastically reducing memory usage while preserving LLM quality.**


---

## 🚀 Welcome to the **official SINQ repository**!
  
**SINQ** (Sinkhorn-Normalized Quantization) is a **novel, fast and high-quality quantization method** designed to make any Large Language Models **smaller** while keeping their accuracy almost intact.

### 🔍 What You’ll Find Here

-  A brief overview of how **SINQ** works under the hood
-  **Why** should I use **SINQ?**
- <u>A few lines of code to <strong>quantize any LLM</strong> with <strong>SINQ</strong>!</u>  
- Code to **reproduce results** from our paper  
- On going updates on new features and integrations (🤗)

#### 📊 Feature Comparison: <u>SINQ vs HQQ</u> _(calibration free)_ and <u>A-SINQ vs AWQ</u> _(calibrated)_


<div align="center">

| Feature | **SINQ** | **HQQ** | │ | **A-SINQ** | **AWQ** |
|--------|:--------:|:--------:|:--:|:----------:|:-------:|
| Calibration | Calibration-free |  Calibration-free | │ | Calibrated | Calibrated |
| Quantization Type | ✅ Symmetric & Asymmetric | ❌ Asymmetric only | │ | ✅ Symmetric & Asymmetric | ✅ Symmetric & Asymmetric |
| NF4 Support | ✅ Yes | ❌ No | │ | ✅ Yes | ❌ No |
| Quantization Speed | ⚡ ~2× Faster | 🐢 Slower | │ | ⚡ ~4× Faster | 🐢 Slower |
| Model Quality | ⭐ Higher | ⚠️ Lower | │ | ⭐ Higher | ⚠️ Lower |

</div>

📄 **Want to know more?** Read our paper on [**arXiv**](http://arxiv.org/abs/2509.22944)!


---

## 🧠 How does SINQ work?

<details>
<summary>Click to expand a quick explanation of SINQ’s core idea</summary>

#### 1️⃣ Dual-Scaling for Better Quantization

<p align="left">
  <img src="imgs/dualscale.png" alt="Dual Scale Illustration" width="330" align="right" style="margin-left: 20px;"/>
</p>

Conventional quantization uses **one scale per weight dimension**, which makes models vulnerable to **outliers**: large weights that distort scaling and cause significant errors.

**SINQ** solves this by introducing **dual scaling**: separate scale factors for **rows and columns**. This flexibility redistributes outlier influence and keeps quantization errors smaller and more balanced.

---


#### 2️⃣ More Even Error Distribution

<p align="left">
  <img src="imgs/error.png" alt="Error Distribution Comparison" width="370" align="right" style="margin-left: 20px;"/>
</p>

With standard single-scale quantization, errors tend to **cluster around outliers**.  
With **SINQ**, they become **spread out and less severe**, preserving model accuracy even at **3 bit precision**. This improvement is driven by SINQ’s **Sinkhorn-normalized optimization**, which iteratively rescales rows and columns to balance their variance - a process inspired by Sinkhorn matrix normalization. By reducing the overall **_matrix imbalance_** (refer to the paper for more info), weights become inherently easier to quantize, leading to more stable behavior across layers and consistently higher accuracy even at very low bit-widths.


</details>

---
## 💡 Why should I use SINQ?
<details>
<summary>Click to expand a quick explanation on why you should use SINQ to quantize your LLM</summary>


#### **SINQ (calibration-free)**  
- **Higher LLM quality** and **~2× faster** quantization than **HQQ** 
- **>31× faster** quantization process and comparable or better LLM quality compared to **AWQ / GPTQ**
- **Model-agnostic**: works without knowing the specific LLM architecture, unlike **QuaRot**  
- **Training-free**: it does not require end-to-end training, unlike **SpinQuant** or **KurTail** 
- **Additionally, A-SINQ (calibrated)** further **beats AWQ, GPTQ, and Hadamard+GPTQ** on quality while achieving **>4× faster** quantization time.

**Example**  
- ⏱️ SINQ quantizes **Qwen3-14B** in just **~21 sec** and **DeepSeekV2.5-236B** in **~5 min** on a single GPU
- 💾 Enables you to **run DeepSeekV2.5-236B** on a single GPU with **~110 GB** of memory (vs ~472 GB) while losing **< 1 ppl** on **WikiText2** and **C4**
</details>

## ⚡ Quantize Any LLM with SINQ

### 1. Setup & Quick Start

First, install the dependencies and set up the package:

```bash
# 1. Clone the repository
git clone https://github.com/huawei-csl/SINQ.git
cd sinq

# 2. Install dependencies
pip install -r req.txt

# 3. Install SINQ
pip install .
```

---

### 2. Quantize in a few lines

Quantizing any 🤗 Hugging Face model with SINQ is simple and takes only a few lines of code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig

model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_cfg = BaseQuantizeConfig(
    nbits=4,            # quantization bit-width
    group_size=128,     # group size
    tiling_mode="1D",   # tiling strategy
    method="sinq"       # quantization method ("asinq" for the calibrated version)
)

AutoSINQHFModel.quantize_model(
    model,
    tokenizer=tokenizer,
    quant_config=quant_cfg,
    compute_dtype=torch.bfloat16,
    device="cuda:0"
)
```

✅ That’s it. Your model is now quantized with **SINQ** and ready for inference or saving.

---

### 3. Optional Flags

You can further customize the quantization process to balance **accuracy** and **memory** for your needs.  
Here’s a summary of the main arguments you can tune:

| Flag | Description | Options | Default |
|------|-------------|---------|----------|
| `--nbits` | Bit-width for weight quantization | 2, 3, 4, 5, 6, 8 | 4 |
| `--tiling_mode` | Weight matrix tiling strategy | 1D, 2D | 1D |
| `--group_size` | Weights per quantization group | 64, 128 | 64 |
| `--method` | Quantization method | sinq, asinq | sinq |


💡 **Tip:** For most cases, the defaults (`--nbits 4 --tiling_mode 1D --group_size 64 --method sinq`) provide an excellent trade-off between compression and accuracy.

### 4. Compatible with [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness) evaluation framework

Below is a minimal example showing how to evaluate a SINQ-quantized model on a benchmark dataset:

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Wrap the already quantized model and tokenizer with HFLM
lm = HFLM(pretrained=model, tokenizer=tokenizer, device="cuda:0")

# Evaluate (many tasks available on lm-eval such as MMLU and HellaSwag)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["lambada_openai"],  # small and fast benchmark
    device="cuda:0"
)
```

## 🧪 How to reproduce paper results
<details>
<summary>Click to expand the commands to reproduce the paper results</summary>

### 1. Setup & Quick Start

First, install the dependencies and set up the package:

```bash
# 1. Clone the repository
git clone https://github.com/huawei-csl/SINQ.git
cd sinq

# 2. Install dependencies
pip install -r req.txt

# 3. Install SINQ
pip install .
```

Then run the following command to quantize **Qwen3-1.7B** out of the box:

```bash
cd tests
python quant_model_eval.py
```

By default, this will run SINQ with the following settings:

- ✅ 4-bit weight quantization  
- ✅ Dual-scale + shift parameterization  
- ✅ 1D tiling  
- ✅ Group size = 64  

---

### 2. Uniform, Uncalibrated Quantization

Reproduce the **core SINQ results** (as shown in Table 1 of the paper):

```bash
python quant_model_eval.py --model_name Qwen/Qwen3-1.7B
```

This uses **INT4 uniform quantization** without calibration - the main benchmark setting of the paper.

---

### 3. Non-Uniform Quantization (NF4)

Try SINQ with **non-uniform quantization** (e.g., NF4):

```bash
python quant_model_eval.py --method sinq_nf4 --model_name Qwen/Qwen3-1.7B
```

---

### 4. Calibrated Quantization (AWQ + SINQ = A-SINQ)

Combine SINQ with **activation-aware calibration (AWQ)** for higher accuracy:

```bash
python quant_model_eval.py --method asinq --model_name Qwen/Qwen3-1.7B
```

---

### ⚙️ Optional Flags

Customize experiments with the following command-line arguments:

| Flag | Description | Options | Default |
|------|-------------|---------|----------|
| `--nbits` | Number of bits used to quantize model weights | 2, 3, 4, 8 | 4 |
| `--tiling_mode` | Strategy for tiling weight matrices during quantization | 1D, 2D | 1D |
| `--group_size` | Number of weights processed together as a quantization group | 64, 128 | 64 |

> 📝 **Note:** All results reported in the paper were obtained using the evaluation framework from [Efficient-ML/Qwen3-Quantization](https://github.com/Efficient-ML/Qwen3-Quantization) rather than `lm-eval`. 
</details>

## 🧭 Ongoing updates on new features and integrations

We are actively expanding SINQ with new features and integrations. Stay tuned here for the latest updates:

- **26/09/2025** - SINQ paper released on [**arXiv**]([https://arxiv.org](http://arxiv.org/abs/2509.22944))  
- **30/09/2025** - SINQ GitHub repository made public  
- 🔜 **Coming soon** – 🤗 Integration with **Hugging Face Transformers**  
- 🔜 **Coming soon** – 📦 Pre-quantized **SINQ models** available on Hugging Face Hub

## 📚 How to Cite This Work

If you find **SINQ** useful in your research or applications, please cite our <a href="http://arxiv.org/abs/2509.22944" target="_blank"><strong>paper</strong></a>:

```bibtex
@misc{muller2025sinq,
      title={SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights}, 
      author={Lorenz K. Muller and Philippe Bich and Jiawei Zhuang and Ahmet Celik and Luca Benfenati and Lukas Cavigelli},
      year={2025},
      eprint={2509.22944},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={http://arxiv.org/abs/2509.22944}
}
```


## 🔗 Related Repositories

This project builds upon and extends the excellent work from the following open-source projects:

- [**Qwen3-Quantization**](https://github.com/Efficient-ML/Qwen3-Quantization) - Base implementation and evaluation scripts for Qwen3 quantization.  
- [**HQQ**](https://github.com/mobiusml/hqq) - High-quality calibration-free quantization baseline.

📜 You can find their original licenses in the corresponding `LICENSE` files in these repositories.








