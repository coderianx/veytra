<div align="center">

# ✦ Veytra ✦

**From words to vectors, from vectors to meaning.**

A lightweight, elegant **sentence embedding model**, built from scratch — trained on semantic textual similarity.

[![Params](https://img.shields.io/badge/params-3.3M-blue)](https://github.com/coderian/veytra)
[![Embedding](https://img.shields.io/badge/embedding-64d-green)](https://github.com/coderian/veytra)
[![Dataset](https://img.shields.io/badge/dataset-STS--B-orange)](https://huggingface.co/datasets/sentence-transformers/stsb)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

</div>

---

## 📖 Overview

**Veytra** is a compact sentence embedding model built entirely from scratch — no pretrained backbone, no fine-tuning shortcuts. It uses a small Transformer Encoder stack to map sentences into a **64-dimensional** vector space, where the **semantic closeness** of any two sentences can be measured with cosine similarity.

It was trained on the [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb) benchmark: given two sentences and a human-annotated similarity score (0–5), the model learns to place semantically equivalent sentences close together in embedding space.

Despite only ~3.3M parameters, Veytra captures meaningful semantic structure — proof that a well-designed small model can go a long way.

## ✨ Features

- 🏗️ **From scratch** — custom Transformer Encoder implemented in pure PyTorch
- ⚡ **Tiny & fast** — ~3.3M parameters, runs comfortably on CPU
- 📏 **Compact embeddings** — fixed 64-dim vectors, L2-normalized for instant cosine similarity
- 🧠 **GPT-2 tokenizer** — robust BPE tokenization out of the box
- 🎯 **STS-trained** — directly optimized against human similarity judgments

## ⚙️ Architecture

| Component | Value |
|---|---|
| **Total Parameters** | ~3.3M (3,316,544) |
| **Tokenizer** | GPT-2 (50,257 vocab) |
| **Model** | Transformer Encoder (2 layers, 4 heads) |
| **Embedding Dimension** | 64 |
| **Feed-Forward Dimension** | 256 |
| **Max Length** | 64 tokens |
| **Positional Encoding** | Sinusoidal |
| **Pooling** | Mean Pooling + L2 Normalization |

The pipeline is simple and clean:

```
Tokens → Embedding → Positional Encoding → Transformer Encoder → Mean Pooling → L2 Norm → 64-dim vector
```

## 🏋️ Training

| Setting | Value |
|---|---|
| **Dataset** | [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb) (`sentence-transformers/stsb`) |
| **Objective** | MSE between cosine similarity and normalized human scores |
| **Score Range** | 0–5 → rescaled to 0–1 |
| **Optimizer** | AdamW |
| **Learning Rate** | 3e-4 |
| **Batch Size** | 32 |
| **Epochs** | 10 |
| **Device** | CUDA if available, otherwise CPU |

For each training pair, both sentences are encoded, their cosine similarity is computed and rescaled to `[0, 1]`, then minimized against the ground-truth score via MSE loss.

## 🚀 Getting Started

### Requirements

- Python 3.11+
- CUDA (optional, for GPU training)

### Installation

```bash
git clone https://github.com/coderian/veytra.git
cd veytra

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Training

```bash
python3 train.py
```

Average loss is printed after every epoch; the trained weights are saved to `veytra-embed-base.pt`.

### Interactive Similarity Test

```bash
python3 test.py
```

Enter two sentences and get an instant similarity report:

```
Sentence 1: The cat sat on the mat
Sentence 2: A cat is sitting on a mat

Cosine similarity : 0.8412
Similarity (0-1)  : 0.9206
STS-B score (0-5) : 4.6031
```

Type `q` to quit.

## 📁 Project Structure

```
veytra/
├── config.py                  # Dataset loading settings
├── train.py                   # Model architecture + training loop
├── test.py                    # Interactive similarity test
├── push_hf.py                 # Hugging Face Hub upload
├── requirements.txt           # Python dependencies
├── sentence_embedding_64d.pt  # Trained model weights
└── venv/                      # Virtual environment
```

## 🤗 Hugging Face

The model is also available on the Hugging Face Hub:

👉 [coderian/veytra-embed-base](https://huggingface.co/coderian/veytra-embed-base)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

*Veytra — encoding meaning.*

</div>
