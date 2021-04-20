import torchaudio
import librosa

# There exist a limitation of this defense that it may lead to the problem of aliasing, and we can use the narrowband sample rate
# rather than downsampling followed by upsampling.
# resampling reference https://core.ac.uk/download/pdf/228298313.pdf
# resampling input transformation defense for audio

T = torchaudio.transforms

# Read audio file
audio_data = librosa.load(files, sr=16000)[0][-19456:]

audio_data = torch.tensor(audio_data).float().to(device)

# Discarding samples from a waveform during downsampling could remove a significant portion of the adversarial perturbation, thereby prevents an adversarial attack.

# resample the audio files to 8kHz from 16kHz
sample = T.Resample(16000, 8000, resampling_method="sinc_interpolation")

audio_resample_1 = sample(audio_data)

# resample the audio back to 16kHz
sample = T.Resample(8000, 16000, resampling_method="sinc_interpolation")

# Give audio_resample_2 as input to the asr model
audio_resample_2 = sample(audio_resample_1)
