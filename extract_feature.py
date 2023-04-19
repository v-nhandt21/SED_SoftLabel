import glob
import config
import torchaudio, torch, tqdm, os
from utils import *

target_sample_rate = 16000

soft_gt_file = open("metadata/soft_gt_dev.csv", "w+", encoding="utf-8")
soft_gt_file.write("filename\tonset\toffset\t" + "\t".join( list(config.class_labels_hard.keys()) )  + "\n")
NO_LABEL = []

for folder in glob.glob(config.data_dir + "development_audio/*"):
     for audio_file in tqdm.tqdm(glob.glob(folder+"/*")):
          
          # Gen Label
          LABEL_DUMP = {}
          label_file = audio_file.replace("development_audio/", "development_annotation/soft_labels_").replace(".wav",".txt")
          labels = open(label_file, "r", encoding="utf-8").read().splitlines()
          for label in labels:
               on, off, lab, soft_score = label.split("\t")
               
               # create torch zero label
               on = int(float(on))
               if on not in LABEL_DUMP:
                    LABEL_DUMP[on] = [0]*(11)
                    
               if lab in config.class_labels_hard:
                    LABEL_DUMP[on][config.class_labels_hard[lab]] = float(soft_score)
               else:
                    if lab not in NO_LABEL:
                         NO_LABEL.append(lab)
          
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
               save_audio_file = audio_file.replace("development_audio", "development_audio_splited").replace(".wav","_"+str(id)+".wav")
               create_folder(os.path.dirname(save_audio_file))
               torchaudio.save(save_audio_file, wave, target_sample_rate)
               
               ################
               save_label_file = audio_file.replace("development_audio", "development_label_splited").replace(".wav","_"+str(id))
               create_folder(os.path.dirname(save_label_file))
               np.save(save_label_file, np.array(LABEL_DUMP[id]))
               
               ################
               soft_gt_file.write(audio_file.split("/")[-1] +"\t"+ str(id) + "\t" + str(id+1) + "\t" + "\t".join([str(s) for s in np.array(LABEL_DUMP[id])]) + "\n")
print(NO_LABEL)
               
          
          