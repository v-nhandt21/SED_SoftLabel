import torch
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import config

MAX_INT = 32767
MIN_INT = -32768
MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
     sampling_rate, data = read(full_path)
     return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
     return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
     return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
     output = dynamic_range_compression_torch(magnitudes)
     return output


def spectral_de_normalize_torch(magnitudes):
     output = dynamic_range_decompression_torch(magnitudes)
     return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y):
     
     center=False
     n_fft=config.n_fft
     num_mels=config.n_mel
     sampling_rate=config.sample_rate
     hop_size=config.hop_size
     win_size=config.win_size
     fmin=50
     fmax=14000
     
     if torch.min(y) < -1.:
          print('min value is ', torch.min(y))
     if torch.max(y) > 1.:
          print('max value is ', torch.max(y))

     global mel_basis, hann_window
     if fmax not in mel_basis:
          mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
          mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
          hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
     y = y.squeeze(1)

     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                         center=center, pad_mode='reflect', normalized=False, onesided=True)

     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

     spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
     spec = spectral_normalize_torch(spec)

     return spec

mel_basis_local = {}
hann_window_local = {}

def mel_spectrogram_local(y):
     
     center=False
     n_fft=1024
     num_mels=64
     sampling_rate=config.sample_rate
     hop_size=78
     win_size=1024
     fmin=0
     fmax=8000
     
     if torch.min(y) < -1.:
          print('min value is ', torch.min(y))
     if torch.max(y) > 1.:
          print('max value is ', torch.max(y))

     global mel_basis_local, hann_window_local
     if fmax not in mel_basis_local:
          mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
          mel_basis_local[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
          hann_window_local[str(y.device)] = torch.hann_window(win_size).to(y.device)

     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
     y = y.squeeze(1)

     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window_local[str(y.device)],
                         center=center, pad_mode='reflect', normalized=False, onesided=True)

     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

     spec = torch.matmul(mel_basis_local[str(fmax)+'_'+str(y.device)], spec)
     spec = spectral_normalize_torch(spec)

     return spec