import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

# Function to load audio and down-sample
def down_sample_audio(audio_original, original_sample_rate):
    print(f"Original Sample Rate: {original_sample_rate}")
    
    target_sample_rate = 16000
    num_samples = int(len(audio_original) * (target_sample_rate / original_sample_rate))

    # Resampling the audio
    downsampledAudio = resample(audio_original, num_samples)

    return downsampledAudio

# Function to process all audio files in a folder
def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Check if the file is a .wav file
        if file_name.endswith(".wav") and os.path.isfile(file_path):
            print(f"Processing file: {file_name}")

            # Load the audio file
            original_sample_rate, audio_original = wavfile.read(file_path)

            # Down-sample the audio
            downsampledAudio = down_sample_audio(audio_original, original_sample_rate)

            # Construct the new file path in the output folder
            new_file_path = os.path.join(output_folder, file_name)

            # Change the file extension if needed (e.g., appending "_downsampled" to the filename)
            new_file_path = os.path.splitext(new_file_path)[0]+".wav"

            # Save the downsampled audio
            wavfile.write(new_file_path, 16000, downsampledAudio.astype(np.int16))  # Ensure correct data type before saving

            print(f"Downsampled audio saved to: {new_file_path}")

# Set the input and output folders
input_folder = r".\OriginalAudio"  # The folder containing the original audio files
output_folder = r".\downSampledAudio"  # The folder where you want to save the downsampled audio

# Process all files in the folder
process_folder(input_folder, output_folder)
