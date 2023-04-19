import os
import numpy as np
import torch, tqdm
import torch.utils.data
from evaluate import *
from utils import split_in_seqs, create_folder
from wavdataset import WavDataset
from model import *

def prediction():
     device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
     SCORE = {}
     
     f_meta = open("Outdir/meta.tsv", "w+", encoding="utf-8")
     f_embedding = open("Outdir/embedding.tsv", "w+", encoding="utf-8")
     
     Tensor_Projector = [0]*11
     
     for fold in config.holdout_fold:
          print("Fold: ", fold)
     
          # Load features and labels
          test_fold = "metadata/development_folds/fold" + str(fold) + "_test.csv"
          
          test_dataset = WavDataset(test_fold)
          
          # Data loader
          test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
          
          if config.mel_input:
               model = CRNN()
          else:
               model = Wav2VecClassifier()
               
          
          
          # model.load_state_dict(torch.load("/home/nhandt23/Desktop/DCASE/Wav2Vec/Outdir/best_fold5.bin", map_location=device))
          model.load_state_dict(torch.load("/home/nhandt23/Desktop/DCASE/Wav2Vec/Outdir/best_fold"+str(fold)+".bin", map_location=device))
          
          model.eval()
          with torch.no_grad():
               for datas, target, files in tqdm.tqdm(test_loader):
                    prediction, embedding = model(datas)
                    
                    label = torch.argmax(target[0], dim=0)
                    if Tensor_Projector[int(label)] < 700:
                         Tensor_Projector[int(label)] = Tensor_Projector[int(label)] + 1
                         
                         label = list(config.class_labels_hard.keys())[int(label)]
                         f_meta.write(str(label) + "\n")
                         emb = embedding[0]
                         f_embedding.write("\t".join([str(s) for s in emb.tolist()]) + "\n")
                    
                    # prediction = torch.nn.functional.softmax(prediction, dim=1)
                    
                    file, on = files[0].split("/")[-1].replace(".wav","").rsplit("_",1)
                    
                    if file not in SCORE:
                         SCORE[file] = {}
                    SCORE[file][int(float(on))] = prediction[0]
     
     for file, timestamp in tqdm.tqdm(SCORE.items()):
          fw = open("dev_txt_scores/" + file + ".tsv", "w+", encoding="utf-8")
          fw.write("onset\toffset\t" + "\t".join( list(config.class_labels_hard.keys()) )  + "\n")
          timestamp = dict(sorted(timestamp.items()))
          for on, prediction in timestamp.items():
               fw.write(str(on) + "\t" + str(int(float(on))+1) + "\t" + "\t".join([str(s) for s in prediction.tolist()]) + "\n")
          fw.close()
          
          
     return 0

def test():

     output_folder = 'dev_txt_scores'
     create_folder(output_folder)

     output_folder_soft = 'dev_txt_scores_soft'
     create_folder(output_folder_soft)
     
     path_groundtruth = 'metadata/gt_dev.csv'
     # Calculate threshold independent metrics
     get_threshold_independent(path_groundtruth, output_folder)
     
     # get_threshold_independent(path_groundtruth, output_folder_soft)


# python train_soft.py 
if __name__ == '__main__':
     # prediction()
     test()
     # get_threshold_independent('metadata/gt_dev.csv', "/home/nhandt23/Desktop/DCASE/Wav2Vec/metadata/dev_txt_scores_gt")