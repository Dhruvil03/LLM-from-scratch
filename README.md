# LLM-from-scratch

A comprehensive implementation of a Large Language Model (LLM) built from the ground up using PyTorch. This project demonstrates the core concepts and architecture behind modern language models like GPT.

## 🎯 Overview

This repository contains a complete implementation of a transformer-based language model, following the principles outlined in the "Attention Is All You Need" paper. The project walks through every component needed to build, train, and use a language model from scratch.

## 🚀 Features

- **Complete Transformer Architecture**: Multi-head attention, feed-forward networks, and positional encoding
- **Tokenization**: Custom tokenizer implementation with BPE (Byte-Pair Encoding)
- **Training Pipeline**: Full training loop with gradient accumulation and learning rate scheduling
- **Text Generation**: Sampling strategies including greedy search, beam search, and nucleus sampling
- **Model Checkpointing**: Save and load model states for inference and continued training
- **Configurable Architecture**: Easy to modify model dimensions, layers, and hyperparameters


## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dhruvil03/LLM-from-scratch.git
   cd LLM-from-scratch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch** (if not included in requirements):
   ```bash
   # For CPU only
   pip install torch torchvision torchaudio
   
   # For CUDA (check your CUDA version)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Generating Text

```python
from src.model.transformer import GPTModel
from src.generation.inference import generate_text

# Load trained model
model = GPTModel.from_pretrained('checkpoints/model.pth')

# Generate text
prompt = "Who is Harry Potter?"
generated_text = generate_text(model, prompt, max_length=100)
print(generated_text)
```

## 🏗️ Architecture Details

### Transformer Components

- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Position Encoding**: Sinusoidal positional embeddings
- **Feed-Forward Networks**: Two-layer MLPs with GELU activation
- **Layer Normalization**: Applied before each sub-layer (Pre-LN)
- **Residual Connections**: Skip connections around each sub-layer

```text
Input text
   │
   ├── Tokenization
   │      text → token IDs
   │
   ├── Token Embedding
   │      Embedding(vocab_size=50257, emb_dim=768)
   │
   ├── Positional Embedding
   │      Embedding(context_length=1024, emb_dim=768)
   │
   ├── Add token + position embeddings
   │
   ├── Dropout(0.1)
   │
   ├── 12 × Transformer Blocks
   │      Each block:
   │
   │      Input x
   │        │
   │        ├── LayerNorm
   │        ├── Masked Multi-Head Self-Attention
   │        │      • 12 heads
   │        │      • causal mask (can only see past tokens)
   │        ├── Dropout
   │        ├── Residual Add
   │        │
   │        ├── LayerNorm
   │        ├── Feed Forward Network
   │        │      Linear(768 → 3072)
   │        │      GELU
   │        │      Linear(3072 → 768)
   │        ├── Dropout
   │        └── Residual Add
   │
   ├── Final LayerNorm
   │
   ├── Output Head
   │      Linear(768 → 50257)
   │
   └── Logits over vocabulary
          ↓
     Next-token prediction
          ↓
     Autoregressive text generation
```

### Model Specifications

| Parameter | Default Value |
|-----------|---------------|
| Model Dimension | 768 |
| Number of Layers | 12 |
| Attention Heads | 12 |
| FFN Dimension | 3072 |
| Vocabulary Size | 50,257 |
| Max Sequence Length | 256 |


## 🔧 Configuration

```python
MODEL_CONFIG = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
```


## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
