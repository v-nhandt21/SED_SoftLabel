import glob
import config
import torchaudio, torch, tqdm, os
from utils import *

target_sample_rate = 16000

data_dir = "/home/nhandt23/Desktop/DCASE/Evaluation/"

for folder in glob.glob(data_dir + "Evaluation_audio/*"):
     for audio_file in tqdm.tqdm(glob.glob(folder+"/*")):
          
          # Gen Audio
          waveform, sr = torchaudio.load(audio_file)
          # print(waveform.size())
          waveform_mono = torch.mean(waveform, dim=0).unsqueeze(0)
          # print(waveform_mono.size())
          waveform_mono_16k = torchaudio.transforms.Resample(sr, target_sample_rate)(waveform_mono)
          # print(waveform_mono_16k.size())
          for id in range(int(waveform_mono_16k.size(1)/target_sample_rate)):
               wave = waveform_mono_16k[:,id*target_sample_rate:(id+1)*target_sample_rate]
               
               ################
               save_audio_file = audio_file.replace("Evaluation_audio", "Evaluation_audio_splited").replace(".wav","_"+str(id)+".wav")
               create_folder(os.path.dirname(save_audio_file))
               torchaudio.save(save_audio_file, wave, target_sample_rate)
          
          