import os
import numpy as np
import torch, tqdm
import torch.utils.data
from evaluate import *
from utils import split_in_seqs, create_folder
from wavdataset import WavDataset
from model import *
import sed_eval, glob

def prediction():
     device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
     # For evaluating the model, only hard labels will be considered (11 classes)
     segment_based_metrics_all_folds = sed_eval.sound_event.SegmentBasedMetrics(
          event_label_list=config.labels_hard,
          time_resolution=1.0
     )
     
     output_model = sorted(glob.glob('Outdir/*'))[-1]
     
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
          
          # model.load_state_dict(torch.load("/home/nhandt23/Desktop/DCASE/Wav2Vec/Outdir/best_fold5.bin", map_location=device))
          model.load_state_dict(torch.load(output_model + "/best_fold"+str(fold)+".bin", map_location=device))
          
          model.eval()
          nbatch = 0
          with torch.no_grad():
               for datas, batch_target, files in tqdm.tqdm(test_loader):
                    prediction, embedding = model(datas)
                    
                    # Feed into the model
                    framewise_output = prediction.detach().cpu().numpy()

                    # Append to evaluate the whole test fold at once
                    if nbatch == 0:
                         fold_target = batch_target
                         fold_output = framewise_output
                    else:
                         fold_target = np.append(fold_target, batch_target, axis=0)
                         fold_output = np.append(fold_output, framewise_output, axis=0)
                    
                    nbatch+=1
                    
               reference = process_event(config.labels_hard, fold_target.T, config.posterior_thresh, 1)

               results = process_event(config.labels_hard, fold_output.T, config.posterior_thresh, 1)

               # Save data for all the folds
               segment_based_metrics_all_folds.evaluate(
                    reference_event_list=reference,
                    estimated_event_list=results
               )

     # End folds

     output = segment_based_metrics_all_folds.result_report_class_wise()
     print(output)
     
     overall_segment_based_metrics_ER = segment_based_metrics_all_folds.overall_error_rate()
     overall_segment_based_metrics_f1 = segment_based_metrics_all_folds.overall_f_measure()
     f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
     er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']
     print(f'\nMicro segment based metrics - ER: {round(er_overall_1sec_list,3)} F1: {round(f1_overall_1sec_list*100,2)} ')
     
     class_wise_metrics = segment_based_metrics_all_folds.results_class_wise_metrics()
     macroFs = []
     for c in class_wise_metrics:
          macroFs.append(class_wise_metrics[c]["f_measure"]["f_measure"])
     
     print(f'\nMacro segment based F1: {round((sum(np.nan_to_num(macroFs))/len(config.labels_hard))*100,2)} ')
     print('\n')


# python train_soft.py 
if __name__ == '__main__':
     prediction()