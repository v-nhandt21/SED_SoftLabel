import librosa
from transformers import Wav2Vec2Processor
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import json, tqdm, glob
import config, os, torchaudio
from pathlib import Path
from utils import split_in_seqs, create_folder, mel_basis, extract_mbe
from augmentation import get_augment_wav, get_augment_mel
from preprocess import mel_spectrogram

class WavDataset(torch.utils.data.Dataset):
     def __init__(self, file_folder, test):
          f = open(file_folder, "r", encoding="utf-8")
          audio_folders = f.read().splitlines()[1:]
          
          self.audios = []
          self.labels = []
          
          for folder in audio_folders:
               files = glob.glob(config.data_dir + "development_audio_splited/" + folder + "_*.wav")
               for file in files:
                    self.audios.append(file)
                    self.labels.append(file.replace("development_audio", "development_label").replace(".wav",".npy"))
                    
          self.specaugment = get_augment_mel()
          self.wavaugment = get_augment_wav
          self.test = test
          
                    
     def __len__(self):
          return len(self.audios)
     
     def get_mel(self, idx):
          audio_file = self.audios[idx]
          label_file = self.labels[idx]

          
          label = np.load(label_file)
          label = torch.from_numpy(label).type(torch.FloatTensor)
          
          audio, _ = torchaudio.load(audio_file) # sr should be 16k
          
          
          if config.wavaugment and not self.test:
               audio = self.wavaugment(audio)
               if config.sample_rate <= audio.size(1):
                    audio = audio[:,:config.sample_rate]
               else:
                    audio = torch.nn.functional.pad(audio, (0, config.sample_rate - audio.size(1),0,0))
          else:
               audio = audio 

          feature_input = mel_spectrogram(audio)[0]
          
          if config.specaugment and not self.test:
               feature_input = self.specaugment(feature_input)
               
          label = label.repeat(5, 1)
               
          return feature_input.T, label, audio_file
     
     def get_mel_chunk(self, idx):
          audio_file = self.audios[idx]
          label_file = self.labels[idx]
          
          onsite = int(float(audio_file.split("_")[-1].split(".")[0]))
          
          audio = None
          label = None
          for i_chunk in range(onsite-int(config.chunk_size/2), onsite+int(config.chunk_size/2) + 1):
               audio_file_cur = audio_file.replace(str(onsite)+".wav", str(i_chunk) + ".wav")
               label_file_cur = label_file.replace(str(onsite)+".npy", str(i_chunk) + ".npy")
               if Path(audio_file_cur).is_file() == True: 
                    audio_cur = torchaudio.load(audio_file_cur)[0]
          
                    label_cur = np.load(label_file_cur)
                    label_cur = torch.from_numpy(label_cur).type(torch.FloatTensor).unsqueeze(0).repeat(int(config.sample_rate/config.hop_size), 1)
                    
               else:
                    audio_cur = torchaudio.load(audio_file)[0]
                    
                    label_cur = np.load(label_file)
                    label_cur = torch.from_numpy(label_cur).type(torch.FloatTensor).unsqueeze(0).repeat(int(config.sample_rate/config.hop_size), 1)
                    
               if audio == None:
                    audio = audio_cur
                    label = label_cur
               else:
                    audio = torch.cat( (audio,audio_cur), dim=1)
                    label = torch.cat( (label,label_cur), dim=0)
          
          if config.wavaugment and not self.test:
               audio = self.wavaugment(audio)
               if config.sample_rate <= audio.size(1):
                    audio = audio[:,:config.sample_rate]
               else:
                    audio = torch.nn.functional.pad(audio, (0, config.sample_rate - audio.size(1),0,0))
          else:
               audio = audio 

          feature_input = mel_spectrogram(audio)[0]
          
          if config.specaugment and not self.test:
               feature_input = self.specaugment(feature_input)
          # print(feature_input.shape)          
          return feature_input.T, label, audio_file
     
     def get_wav(self, idx):
          audio_file = self.audios[idx]
          label_file = self.labels[idx]
               
          label = np.load(label_file)
          label = torch.from_numpy(label).type(torch.FloatTensor)
          
          audio, _ = librosa.load(audio_file, sr = 16000)
          
          if config.wavaugment:
               audio = self.wavaugment(audio)
               
          feature_input = audio

          return feature_input, label, audio_file

     def __getitem__(self, idx):
          
          if config.model == "CRNN":
               return self.get_mel(idx)
          elif config.model == "Wav2VecClassifier":
               return self.get_wav(idx)
          elif config.model == "CRNN_Chunk":
               return self.get_mel_chunk(idx)
     
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
     ds = WavDataset("/home/nhandt23/Desktop/DCASE/DCASE_Sound_Event_Detection_Soft_Labels/development_folds/fold1_train.csv", test=False)
     
     # ids = ds.get_class_idxs()
     # tt = 0
     # for i in ids:
     #      print(len(i))
     #      tt += len(i)
     # print(tt)
     
     dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
     for batch in dl:
          audio, label, audio_file = batch 
          print(audio.shape)
          print(label.shape)
          break
     # processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
     # for audio, label, filename in dl:
     #      print(len(audio[0]), len(audio[1]))
          # inputs = processor(audio, sampling_rate=16000, return_tensors = "pt", padding = True).input_values
          # print(inputs.shape)
          # break
