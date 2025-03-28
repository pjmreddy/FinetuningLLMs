# Whisper Fine-tuning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A specialized implementation for fine-tuning OpenAI's Whisper speech recognition model on custom audio datasets using LoRA (Low-Rank Adaptation) for efficient training.

## 🎯 Project Overview

This project provides a complete pipeline for fine-tuning the Whisper speech recognition model on your custom audio data. It includes:

- Automatic audio preprocessing with sample rate conversion to 16kHz
- Efficient dataset creation from audio files and transcriptions
- Memory-efficient fine-tuning using LoRA adaptation
- Comprehensive training monitoring and checkpointing

## 🚀 Key Features

- **Automated Audio Processing**: Converts various audio formats to 16kHz WAV using scipy
- **Smart Dataset Creation**: Automatically pairs audio files with transcriptions
- **LoRA Fine-tuning**: Uses PEFT library for memory-efficient model adaptation
- **Flexible Training**: Supports various Whisper model sizes (default: whisper-small)
- **Progress Monitoring**: Training logs and metrics via TensorBoard

## 📋 Requirements

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- Datasets >= 2.14.0
- PEFT >= 0.6.0
- SciPy (for audio processing)
- TensorBoard (for monitoring)

## 📁 Project Components

```
.
├── ftwmodel.py        # Main fine-tuning script with LoRA implementation
├── createDataset.py   # Handles dataset creation and preprocessing
├── downsample.py      # Audio conversion to 16kHz WAV format
└── requirements.txt   # Project dependencies
```

## 🎵 Dataset Preparation

1. **Organize Your Data**:
   - Place audio files in `downSampledAudio/`
   - Add corresponding transcriptions in `Transcriptions/`
   - Each audio file should have a matching .txt file (e.g., audio1.wav → audio1.txt)

2. **Audio Processing**:
   - The `downsample.py` script automatically:
     - Converts audio to 16kHz WAV format
     - Ensures consistent audio format for training
     - Handles various input formats (WAV, MP3, etc.)

3. **Dataset Creation**:
   - `createDataset.py` automatically:
     - Matches audio files with transcriptions
     - Creates a CSV dataset file
     - Validates data integrity

## 🔧 Model Fine-tuning

### LoRA Configuration

The project uses optimized LoRA settings for efficient fine-tuning:

```python
lora_config = {
    "r": 16,              # Rank for adaptation
    "lora_alpha": 32,     # Scaling factor
    "lora_dropout": 0.1,  # Dropout rate
    "target_modules": ["q_proj", "k_proj", "v_proj"]
}
```

### Training Process

1. **Run Training**:
```bash
python ftwmodel.py
```

2. **Training Steps**:
   - Loads and preprocesses the dataset
   - Applies LoRA configuration
   - Fine-tunes the model
   - Saves checkpoints and logs

3. **Monitoring**:
   - Training progress in terminal
   - Detailed metrics in TensorBoard
   - Regular model checkpoints

### Output

- **Model Checkpoints**: Saved in `results/`
- **Training Logs**: Available in `logs/`
- **TensorBoard Events**: For visualizing metrics

## 🔍 Performance Monitoring

- Monitor training progress using TensorBoard:
```bash
tensorboard --logdir=logs
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

If you have any questions or suggestions, please feel free to open an issue.

---

*Note: This project is built on top of OpenAI's Whisper model. Please ensure you comply with OpenAI's usage terms and conditions.*