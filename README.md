# 🧠 Fine-Tuning LLMs with PEFT

PyTorch | Hugging Face | PEFT

A comprehensive repo for parameter-efficient fine-tuning (PEFT) of large language models using lightweight methods like LoRA, Adapters, and Prompt Tuning. Designed for minimal compute, high efficiency, and fast iteration.

🚀 Key Features

🔧 PEFT Techniques
LoRA (Low-Rank Adaptation), Adapter Layers, Prefix Tuning, Prompt Tuning

🧬 Model Support
✅ GPT | Whisper | LLaMA | BERT | DistilBERT | TinyBERT | MobileBERT

📂 Tasks Covered
Sentiment Analysis
Named Entity Recognition (NER)
Text Classification
📈 Efficiency First
Achieve 90–95% of full fine-tuned model performance
using <5% of total parameters

🧪 Reproducibility
Built with 🤗 Hugging Face transformers, peft, accelerate
📚 Techniques Covered

Method	Use Case	Example Models
LoRA	Low-rank decomposition	LLaMA-2, BERT variants
Adapters	Lightweight fine-tuning	DistilBERT, TinyBERT
Prefix Tuning	Prompt optimization	GPT-style architectures

⚡ Quick Start

# 1. Install dependencies
pip install torch transformers peft accelerate datasets

# 2. Clone the repo
git clone https://github.com/pjmreddy/FinetuningLLMs
cd FinetuningLLMs

# 3. Run example script
python finetune_sentiment.py --model_name bert-base-uncased --method lora
🧠 Example Notebooks

✅ Finetuning_LLaMA2.ipynb
✅ NER_using_DistilBERT.ipynb
✅ Sentiment_Classification_BERT.ipynb
✅ Comparing_DistilBERT_TinyBERT_MobileBERT.ipynb
📍 Trained on

LONI HPC Clusters – High-performance compute optimized for large model training & experimentation.

Let me know if you want to generate badges, add model performance tables, or integrate Colab links!
