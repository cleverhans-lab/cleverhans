import torchaudio
import librosa

# resampling reference https://core.ac.uk/download/pdf/228298313.pdf
# resampling input transformation defense for audio

T = torchaudio.transforms
audio_data = librosa.load(files, sr=16000)[0][-19456:]  # Read audio file
audio_data = torch.tensor(audio_data).float().to(device)
sample = T.Resample(
    16000, 8000, resampling_method="sinc_interpolation"
)  # resample the audio files to 8kHz from 16kHz
audio_resample_1 = sample(audio_data)
sample = T.Resample(
    8000, 16000, resampling_method="sinc_interpolation"
)  # resample the audio back to 16kHz
audio_resample_2 = sample(audio_resample_1)
# Give audio_resample_2 as input to the asr model
