<div align="center">

<img src="https://img.shields.io/badge/Model-Qwen3--8B-blue?style=for-the-badge&logo=huggingface&logoColor=white" alt="Model">
<img src="https://img.shields.io/badge/Method-LoRA_PEFT-orange?style=for-the-badge" alt="Method">
<img src="https://img.shields.io/badge/Dataset-Gaming_500H-green?style=for-the-badge" alt="Dataset">
<img src="https://img.shields.io/badge/GPU-NVIDIA_T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU">
<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">

<br><br>

# 🎮 Hexar AI Model

### Gaming-Focused AI Assistant — Fine-Tuned Qwen3-8B

<p>
  <b>Hexar</b> is a domain-specific large language model built by fine-tuning
  <a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B</a> on a curated
  gaming dataset using <b>LoRA</b> (Low-Rank Adaptation) for parameter-efficient training.
  The result is a lightweight yet powerful assistant that understands gaming terminology,
  strategies, and community knowledge.
</p>

<img src="https://img.shields.io/badge/🟢_Status-Ready_to_Use-brightgreen?style=flat-square" alt="Status">

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture & Methodology](#-architecture--methodology)
- [Dataset](#-dataset)
- [Training Configuration](#-training-configuration)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Inference](#-model-inference)
- [Hardware Requirements](#-hardware-requirements)
- [Project Structure](#-project-structure)
- [Key Dependencies](#-key-dependencies)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🌟 Overview

Hexar AI is a specialized conversational model designed for the gaming community. By leveraging the powerful Qwen3-8B base model and fine-tuning it with 500 hours of gaming-related content, Hexar delivers accurate, context-aware responses to gaming queries — from game mechanics and strategies to lore discussions and community trends.

The entire training pipeline is optimized to run on a single **NVIDIA T4 GPU** (16 GB VRAM) using **4-bit quantization** and **LoRA**, making it accessible for researchers and developers without access to expensive multi-GPU clusters.

### ✨ Key Highlights

| Feature | Detail |
|---------|--------|
| **Base Model** | Qwen3-8B |
| **Fine-Tuning Method** | LoRA (r=64, alpha=16) |
| **Quantization** | 4-bit NF4 |
| **Training Dataset** | Gaming 500-Hour Dataset |
| **Target GPU** | NVIDIA T4 (16 GB) |
| **Precision** | BFloat16 / FP16 (auto) |
| **Optimizer** | Paged AdamW 32-bit |

---

## 🏗 Architecture & Methodology

Hexar's training pipeline follows a modern parameter-efficient fine-tuning approach:

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Gaming Dataset  │ ──▶  │  Format & Tokenize │ ──▶  │  Qwen3-8B (4-bit)   │
│  (500 Hours)     │      │  (SFT Prompt)     │      │  + LoRA Adapters    │
└─────────────────┘      └──────────────────┘      └─────────┬───────────┘
                                                              │
                                                          Training
                                                              │
                                                              ▼
                                                 ┌─────────────────────┐
                                                 │  Hexar Gaming Model  │
                                                 │  (LoRA Weights Only) │
                                                 └─────────────────────┘
```

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 64 | Rank of the low-rank decomposition |
| `lora_alpha` | 16 | Scaling factor for LoRA updates |
| `lora_dropout` | 0.1 | Dropout probability for LoRA layers |
| `bias` | none | No bias parameters in LoRA layers |
| `task_type` | CAUSAL_LM | Causal language modeling objective |

### Quantization (BitsAndBytes 4-bit)

The model is loaded with **NF4 (NormalFloat 4-bit)** quantization to dramatically reduce memory usage while preserving model quality. This allows the entire 8-billion parameter model to fit within the T4's 16 GB VRAM budget alongside the training overhead.

---

## 📊 Dataset

The model is trained on the **`markov-ai/gaming-500-hours`** dataset, a comprehensive collection of gaming-related conversations and content covering:

- Game strategies and walkthroughs
- Community discussions and opinions
- Gaming terminology and jargon
- Game reviews and recommendations
- Esports commentary and analysis

### Prompt Format

Each training example is formatted into an instruction-following template:

```
### System:
You are Hexar, a helpful AI assistant focused on gaming.

### Instruction:
{title from dataset}

### Response:
{description from dataset}
```

This structured prompt format ensures the model learns consistent conversational behavior aligned with the Hexar persona.

---

## ⚙ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1 |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 1 |
| Learning Rate | 2e-4 |
| Weight Decay | 0.001 |
| Max Gradient Norm | 0.3 |
| Warmup Steps | 12 |
| LR Scheduler | Constant |
| Optimizer | Paged AdamW 32-bit |
| Save Checkpoints | Every 100 steps |
| Logging | Every 10 steps |
| Mixed Precision | BF16 (preferred) / FP16 (fallback) |

> **Note:** The trainer automatically selects BFloat16 if the GPU supports it; otherwise, it falls back to FP16 for maximum compatibility.

---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
# Clone the project
git clone https://github.com/your-username/hexar-model.git
cd hexar-model

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -U transformers accelerate peft bitsandbytes trl datasets torch
```

<details>
<summary>🔧 PyTorch with CUDA (if not pre-installed)</summary>

```bash
# Visit https://pytorch.org/get-started/locally/ for the right command for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

</details>

---

## 🚀 Quick Start

### 1. Fine-Tune the Model

Open `HexarModel.ipynb` in Jupyter, Google Colab, or Kaggle and run all cells sequentially. The notebook handles everything from dataset loading to model training and saving.

For **Google Colab** (recommended):
1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells

For **Kaggle**:
1. Create a new notebook on [Kaggle](https://www.kaggle.com)
2. Enable GPU accelerator in Settings
3. Paste the code and run

### 2. Load the Fine-Tuned Model

After training completes, the LoRA-adapted model is saved to `qwen3-8b-Hexar-Gaming-Model/`. You can load and use it immediately:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("qwen3-8b-Hexar-Gaming-Model")
model = AutoModelForCausalLM.from_pretrained("qwen3-8b-Hexar-Gaming-Model")

# Chat with Hexar
messages = [{"role": "user", "content": "What are the best strategies for Valorant?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(response)
```

---

## 💬 Model Inference

Hexar responds to gaming-related questions with domain-specific knowledge. Example interactions:

```
User: Who are you?
Hexar: I am Hexar, a helpful AI assistant focused on gaming.

User: What's the meta for League of Legends right now?
Hexar: [Provides up-to-date gaming knowledge from training data]

User: How do I improve my aim in FPS games?
Hexar: [Offers gaming-specific tips and strategies]
```

<details>
<summary>📖 Advanced: Merging LoRA Weights</summary>

To merge the LoRA adapters into the base model for deployment:

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "qwen3-8b-Hexar-Gaming-Model")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("hexar-merged")
tokenizer.save_pretrained("hexar-merged")
```

</details>

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA T4 (16 GB) | NVIDIA V100 / A100 |
| **VRAM** | 16 GB | 24 GB+ |
| **RAM** | 16 GB | 32 GB+ |
| **Disk Space** | ~20 GB | ~40 GB |
| **Platform** | Google Colab / Kaggle | Local GPU / Cloud VM |

> The pipeline is designed to work within the memory constraints of a free Google Colab T4 GPU. No multi-GPU setup is required.

---

## 📁 Project Structure

```
hexar-model/
├── HexarModel.ipynb          # Main training notebook
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies
├── qwen3-8b-Hexar-Gaming-Model/   # Output: Fine-tuned LoRA weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── results/                   # Training checkpoints & logs
    └── ...
```

---

## 🔑 Key Dependencies

| Library | Purpose |
|---------|---------|
| [Transformers](https://github.com/huggingface/transformers) | Model loading, tokenization, and inference |
| [PEFT](https://github.com/huggingface/peft) | LoRA parameter-efficient fine-tuning |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization for memory efficiency |
| [TRL](https://github.com/huggingface/trl) | SFTTrainer for supervised fine-tuning |
| [Datasets](https://github.com/huggingface/datasets) | Dataset loading and preprocessing |
| [Accelerate](https://github.com/huggingface/accelerate) | Distributed training and mixed precision |

---

## 🗺 Roadmap

- [x] Initial fine-tuning pipeline on gaming dataset
- [x] LoRA adapter training with 4-bit quantization
- [x] Inference testing and validation
- [ ] Expand training dataset with multi-language gaming content
- [ ] Add RLHF (Reinforcement Learning from Human Feedback)
- [ ] Deploy as an API endpoint
- [ ] Integrate with Discord / Telegram bots
- [ ] Publish on Hugging Face Hub
- [ ] Benchmark against base Qwen3-8B on gaming tasks

---

<div align="center">

**Built with ❤️ for the Gaming Community**

<img src="https://img.shields.io/badge/Made_with-Python-blue?style=flat-square&logo=python" alt="Python">
<img src="https://img.shields.io/badge/Powered_by-Hugging_Face-yellow?style=flat-square&logo=huggingface" alt="HuggingFace">

</div>
