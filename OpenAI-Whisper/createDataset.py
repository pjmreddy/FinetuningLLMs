import os
import pandas as pd

audio_folder = './downSampledAudio'
transcript_folder = './Transcriptions'
Total = 0

# Initialize lists to store the file paths and transcriptions
audio_paths = []
transcriptions = []

# Loop through the audio folder and find corresponding transcript files
for audio_file in os.listdir(audio_folder):
    # Check if the file is a valid audio file (assuming audio files have '.wav' extension)
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(audio_folder, audio_file)
        
        # Get the name of the transcript by replacing the audio extension with '.txt'
        transcript_filename = audio_file.replace('.wav', '.txt')
        transcript_path = os.path.join(transcript_folder, transcript_filename)
        
        # Check if the transcript file exists
        if os.path.exists(transcript_path):
            try:
                # Read the transcription text from the file using UTF-8 encoding
                with open(transcript_path, 'r', encoding='utf-8') as file:
                    transcription = file.read().strip()
                
                # Append the audio file path and its corresponding transcription
                audio_paths.append(audio_path)
                transcriptions.append(transcription)
                Total += 1

            except UnicodeDecodeError as e:
                print(f"Error reading transcript {transcript_path}: {e}")
        else:
            print(f"Transcript not found for audio: {audio_file}")

# Create a DataFrame from the lists
data = {
    'audio_file': audio_paths,
    'transcription': transcriptions
}

df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
output_file = 'Dataset.csv'
df.to_csv(output_file, index=False)  # index=False to avoid writing row numbers

print(f"CSV file created: {output_file}")
print(f"\nTotal audio and transcript files matched: {Total}")
