import librosa
from transformers import Wav2Vec2Processor
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import json, tqdm, glob
import config, os
from pathlib import Path
from utils import split_in_seqs, create_folder


mel_basis = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mel, fmin=50, fmax=14000)

def extract_mbe(_y):
     spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=config.n_fft, hop_length=config.hop_size, power=1)
     return np.dot(mel_basis, spec)

class WavDataset(torch.utils.data.Dataset):
     def __init__(self, file_folder):
          f = open(file_folder, "r", encoding="utf-8")
          audio_folders = f.read().splitlines()[1:]
          
          self.audios = []
          self.labels = []
          
          for folder in audio_folders:
               files = glob.glob(config.data_dir + "development_audio_splited/" + folder + "_*.wav")
               for file in files:
                    self.audios.append(file)
                    self.labels.append(file.replace("development_audio", "development_label").replace(".wav",".npy"))
          
                    
     def __len__(self):
          return len(self.audios)

     def __getitem__(self, idx):
          audio_file = self.audios[idx]
          label_file = self.labels[idx]

          
          label = np.load(label_file)
          label = torch.from_numpy(label).type(torch.FloatTensor)
          
          if config.mel_input:
               mel_file = audio_file.replace("development_audio", "development_mel").replace(".wav",".npy")
               if Path(mel_file).is_file() == False:
                    audio, _ = librosa.load(audio_file, sr = 16000)
                    create_folder(os.path.dirname(mel_file))
                    feature_input = extract_mbe(audio).T
                    np.save(mel_file, feature_input)
               feature_input = np.load(mel_file)
          else:
               audio, _ = librosa.load(audio_file, sr = 16000)
               feature_input = audio

          return feature_input, label, audio_file
     
     def get_class_idxs(self):
          class_idxs = {}
          for idx in range(len(self.audios)):
               label_file = self.labels[idx]
               label = np.load(label_file)
               label = torch.from_numpy(label).type(torch.FloatTensor)
               label = torch.argmax(label, dim=0)
               label = list(config.class_labels_hard.keys())[int(label)]

               if label in class_idxs:
                    class_idxs[label].append(idx)
               else:
                    class_idxs[label] = [idx]
          idxs_list = []
          for k in sorted(class_idxs):
               idxs_list.append(class_idxs[k])
          return idxs_list

if __name__ == "__main__":


     #############################
     ds = WavDataset("/home/nhandt23/Desktop/DCASE/DCASE_Sound_Event_Detection_Soft_Labels/development_folds/fold1_train.csv")
     
     ids = ds.get_class_idxs()
     tt = 0
     for i in ids:
          print(len(i))
          tt += len(i)
     print(tt)
     # dl = DataLoader(dataset=ds, batch_size=8, shuffle=True,
     #                                                   num_workers=1, pin_memory=True, drop_last=True)
     # for batch in dl:
     #      audio, label, audio_file = batch 
     #      print(audio.shape)
     # processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
     # for audio, label, filename in dl:
     #      print(len(audio[0]), len(audio[1]))
          # inputs = processor(audio, sampling_rate=16000, return_tensors = "pt", padding = True).input_values
          # print(inputs.shape)
          # break