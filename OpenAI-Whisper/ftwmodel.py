# !pip install transformers peft
# !sudo apt-get install ffmpeg
# !pip install datasets torchaudio librosa transformers accelerate
# !pip install torch torchaudio transformers datasets peft accelerate bitsandbytes jiwer

import os
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig

# Define paths
audio_path = "./downSampledAudio"
transcription_path = "./Transcriptions"


# Get list of audio files
audio_files = sorted([f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))])

# Match audio files with transcription files
data = []
for audio_file in audio_files:
    transcription_file = os.path.splitext(audio_file)[0] + ".txt"
    transcription_filepath = os.path.join(transcription_path, transcription_file)
    if os.path.exists(transcription_filepath):
        with open(transcription_filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        data.append({"audio_path": os.path.join(audio_path, audio_file), "text": text})

# Convert to DataFrame and save CSV
df = pd.DataFrame(data)
csv_path = "./dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset saved at {csv_path}")


# Load dataset from CSV
df = pd.read_csv(csv_path)


# Split dataset: First 5 samples for training, all for testing
train_df = df.iloc[:5]
test_df = df
train_df.to_csv("./train.csv", index=False)
test_df.to_csv("./test.csv", index=False)
print("Train and test sets are ready!")
print("\nTrain Set:\n", train_df)
print("\nTest Set:\n", test_df)

# Function to convert audio to 16kHz WAV
def convert_to_wav_16k(audio_path, output_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert only if sample rate is not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    # Save as WAV
    torchaudio.save(output_path, waveform, 16000)


# Process train and test audio files to ensure they are 16kHz WAV
for csv_file in ["train.csv", "test.csv"]:
    df_csv = pd.read_csv(f"./{csv_file}")
    for i, row in df_csv.iterrows():
        input_audio = row["audio_path"]
        # Adjust extension if necessary
        output_audio = input_audio.replace(".mp3", ".wav").replace(".flac", ".wav")
        convert_to_wav_16k(input_audio, output_audio)

# Function to load dataset from CSV into HuggingFace Dataset
def load_whisper_dataset(csv_path):
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    return dataset

# Load train and test datasets
train_dataset = load_whisper_dataset("./train.csv")
test_dataset = load_whisper_dataset("./test.csv")
print(train_dataset)
print(test_dataset)

# Load Whisper model and processor
model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# LoRA configuration for fine-tuning
lora_config = LoraConfig(
    r=16,                # Rank for LoRA
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj"],  # Target modules for LoRA
    bias="none"
)
model = get_peft_model(model, lora_config)

# (Re)load the processor if needed
processor = WhisperProcessor.from_pretrained(model_name)

# Preprocessing function for audio and text
def preprocess_function(examples):
    # Load the audio file from its path
    audio_input, _ = torchaudio.load(examples["audio_path"])
    audio_input = audio_input.mean(dim=0)  # Convert stereo to mono
    # Resample the audio to 16kHz (if not already)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000)
    audio_input = resampler(audio_input)
    # Process audio using the WhisperProcessor
    audio_features = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    # Tokenize the transcription text with truncation
    labels = processor.tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=448, 
        return_tensors="pt"
    ).input_ids
    # Return both input features and labels
    return {
        "input_features": audio_features["input_features"].squeeze().tolist(),
        "labels": labels.squeeze().tolist()
    }

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio_path", "text"])
test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio_path", "text"])

# Set the format to PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_features", "labels"])
test_dataset.set_format(type="torch", columns=["input_features", "labels"])

# (Optional) Reload the model and processor for training if needed
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Define a custom data collator for Seq2Seq tasks using the feature extractor for padding
class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = processor.feature_extractor.pad(features, padding=True, return_tensors="pt")
        return batch

data_collator = CustomDataCollatorForSeq2Seq(processor)

# Define training arguments with a save strategy that saves at the end of each epoch
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save model checkpoints
    eval_strategy="epoch",          # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save checkpoint at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjusted batch size for small dataset
    per_device_eval_batch_size=2,
    num_train_epochs=1,             # Number of training epochs
    weight_decay=0.01,
    logging_dir="./logs",           # Logging directory
    logging_steps=10,
    save_total_limit=2,             # Limit total saved checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,  # Using feature_extractor here
)

# Start training the model
print("Training the model...")
trainer.train()

# Save the final model manually (in case no checkpoint was saved during training)
trainer.save_model("./results/final_model")
print("Training complete!")

# Evaluate the model
print("Evaluating the model...")
trainer.evaluate()

# Example of a prediction function
def predict(example):
    inputs = processor(example["input_features"], return_tensors="pt", sampling_rate=16000)
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

# Apply prediction to a sample test item
sample = test_dataset[0]
predicted_transcription = predict(sample)
print("Predicted transcription:", predicted_transcription)
print("Test sample keys:", list(sample.keys()))
