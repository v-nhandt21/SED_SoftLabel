import os
import numpy as np
import torch, tqdm
import torch.utils.data
from evaluate import *
from utils import split_in_seqs, create_folder
from wavdataset import WavDataset
from model import *
import glob

def prediction(output_model):
     device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
     SCORE = {}
     
     f_meta = open(output_model + "/meta.tsv", "w+", encoding="utf-8")
     f_meta_all = open(output_model + "/meta_all.tsv", "w+", encoding="utf-8")
     f_embedding = open(output_model + "/embedding.tsv", "w+", encoding="utf-8")
     
     Tensor_Projector = [0]*11
     
     frame_sec = int(config.sample_rate/config.hop_size)
     
     for fold in config.holdout_fold:
          print("Fold: ", fold)
     
          # Load features and labels
          test_fold = "metadata/development_folds/fold" + str(fold) + "_test.csv"
          
          test_dataset = WavDataset(test_fold, test=True)
          
          # Data loader
          test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
          
          if config.model == "CRNN" or config.model == "CRNN_Chunk":
               model = CRNN()
          else:
               model = Wav2VecClassifier()

          # model.load_state_dict(torch.load(output_model + "/best_fold5.bin", map_location=device))
          model.load_state_dict(torch.load(output_model + "/best_fold"+str(fold)+".bin", map_location=device))
          model.eval()
          with torch.no_grad():
               for datas, target, files in tqdm.tqdm(test_loader):
                    prediction, embedding = model(datas)
                    
                    prediction = prediction[:,int(config.chunk_size/2)*frame_sec:(int(config.chunk_size/2)+1)*frame_sec,:]
                    
                    file, on = files[0].split("/")[-1].replace(".wav","").rsplit("_",1)
                    
                    if file not in SCORE:
                         SCORE[file] = {}
                    for idx, pre in enumerate(prediction[0]):
                         SCORE[file][int(float(on)) + round(idx*(1/frame_sec),1)] = prediction[0][idx]
                    
     create_folder(output_model+"/dev_txt_scores")
     for file, timestamp in tqdm.tqdm(SCORE.items()):
          fw = open(output_model+"/dev_txt_scores/" + file + ".tsv", "w+", encoding="utf-8")
          fw.write("onset\toffset\t" + "\t".join( list(config.class_labels_hard.keys()) )  + "\n")
          timestamp = dict(sorted(timestamp.items()))
          for on, prediction in timestamp.items():
               fw.write(str(round(on,1)) + "\t" + str(round(on+(1/frame_sec),1)) + "\t" + "\t".join([str(s) for s in prediction.tolist()]) + "\n")
          fw.close()
          
          
     return 0

def test(output_model):
     
     path_groundtruth = 'metadata/gt_dev.csv'
     # Calculate threshold independent metrics
     get_threshold_independent(path_groundtruth, output_model + '/dev_txt_scores')
     
     # get_threshold_independent(path_groundtruth, output_folder_soft)


# python train_soft.py 
if __name__ == '__main__':
     
     
     output_model = sorted(glob.glob('Outdir/*'))[-1]
     
     
     print(output_model)
     
     # output_model = 'Outdir/20230420141731'
     prediction(output_model)
     test(output_model)
     # get_threshold_independent('metadata/gt_dev.csv', "/home/nhandt23/Desktop/DCASE/Wav2Vec/metadata/dev_txt_scores_gt")
