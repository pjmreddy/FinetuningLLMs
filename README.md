# 🧠 Fine-Tuning LLMs with PEFT

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-0.5%2B-blue)](https://github.com/huggingface/peft)

A repository demonstrating **parameter-efficient fine-tuning** of Large Language Models (LLMs) using techniques like **LoRA**, **Adapters**, and **Prompt Tuning**. Optimized for minimal compute and rapid experimentation.

---

## 🚀 Key Features
- **PEFT Methods**: LoRA (Low-Rank Adaptation), Adapter Layers, Prefix Tuning  
- **Supported Models**: GPT, Whisper, LLaMA, BERT, DistilBERT, TinyBERT, MobileBERT
- **Tasks**: Sentiment Analysis, Named Entity Recognition (NER), Text Classification  
- **Efficiency**: Achieve >90% baseline performance with <5% tuned parameters  
- **Reproducibility**: Hugging Face `transformers` + `peft` integration  

---

## 📚 Techniques Covered
| **Method**       | **Use Case**           | **Example Models**       |  
|-------------------|------------------------|--------------------------|  
| **LoRA**          | Low-rank decomposition | LLaMA-2, BERT variants   |  
| **Adapters**      | Task-specific layers   | DistilBERT, TinyBERT     |  
| **Prefix Tuning** | Prompt optimization    | GPT-style architectures  |  

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers peft accelerate datasets
