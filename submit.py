import os
import numpy as np
import torch, tqdm
import torch.utils.data
from evaluate import *
from utils import split_in_seqs, create_folder
from wavdataset import WavDataset
from model import *
import glob
import shutil

def prediction(output_model, fold):
     device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
     SCORE = {}
     
     
     

     print(config.model)
     
     frame_sec = int(config.sample_rate/config.hop_size)

     # Load features and labels
     test_fold = "/home/nhandt23/Desktop/DCASE/Evaluation/meta.csv"
     
     test_dataset = WavDataset(test_fold, test=True, submit=True)
     
     # Data loader
     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
     
     if config.model == "CRNN" or config.model == "CRNN_Chunk":
          model = CRNN_Chunk()
     elif config.model == "CRNN_GLA":
          model = CRNN_GLA()
     else:
          model = Wav2VecClassifier()

     # model.load_state_dict(torch.load(output_model + "/best_fold5.bin", map_location=device))
     model.load_state_dict(torch.load(output_model + "/best_fold"+str(fold)+".bin", map_location=device))
     model.eval()
     with torch.no_grad():
          for datas, target, files in tqdm.tqdm(test_loader):
               
               prediction, embedding = model(datas)
               
               if config.model != "CRNN_GLA":
                    prediction = prediction[:,int(config.chunk_size/2)*frame_sec:(int(config.chunk_size/2)+1)*frame_sec,:]
               
               if config.model == "CRNN_GLA":
                    prediction, local_prediction = prediction
                    prediction = prediction[:,int(config.chunk_size/2)*frame_sec:(int(config.chunk_size/2)+1)*frame_sec,:]
               
               prediction = torch.nn.functional.softmax(prediction, dim=2)
               
               file, on = files[0].split("/")[-1].replace(".wav","").rsplit("_",1)
               
               if file not in SCORE:
                    SCORE[file] = {}
               for idx, pre in enumerate(prediction[0]):
                    SCORE[file][int(float(on)) + round(idx*(1/frame_sec),1)] = prediction[0][idx]
                         
                         
                         
                    
     create_folder(output_model+"/submit")
     for file, timestamp in tqdm.tqdm(SCORE.items()):
          fw = open(output_model+"/submit/" + file + ".tsv", "w+", encoding="utf-8")
          fw.write("onset\toffset\t" + "\t".join( list(config.class_labels_hard.keys()) )  + "\n")
          timestamp = dict(sorted(timestamp.items()))
          for on, prediction in timestamp.items():
               fw.write(str(round(on,1)) + "\t" + str(round(on+(1/frame_sec),1)) + "\t" + "\t".join([str(s) for s in prediction.tolist()]) + "\n")
          fw.close()
          
          
     return 0

if __name__ == '__main__':
     
     
     output_model = sorted(glob.glob('Outdir/*'))[-1]
     print(output_model.split("/")[-1])
     
     
     output_model = "Outdir/20230508110052" # 46.7%

     fold=3
     
     prediction(output_model, fold)