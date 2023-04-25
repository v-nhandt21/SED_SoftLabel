import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from evaluate import *
from model import *
from utils import *
import config, json
import sed_eval, tqdm
from wavdataset import WavDataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pytorch_balanced_sampler.sampler import SamplerFactory
import shutil
import warnings
warnings.filterwarnings("ignore")

def train():
     device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

     # Create output folders
     output_model = 'Outdir/' + get_time()
     create_folder(output_model)
     shutil.copy('config.py', output_model)

     output_folder = 'dev_txt_scores'
     create_folder(output_folder)

     output_folder_soft = 'dev_txt_scores_soft'
     create_folder(output_folder_soft)
     
     LogWanDB = False #a.use_wandb
     if LogWanDB:
          import wandb
          wandb.init(sync_tensorboard=True)
     
     epoch_global = 0

     for fold in config.holdout_fold:
          
          print("Fold: ", fold)

          # Load features and labels
          train_fold = "metadata/development_folds/fold" + str(fold) + "_train.csv"
          val_fold = "metadata/development_folds/fold" + str(fold) + "_val.csv"
          
          train_dataset = WavDataset(train_fold, test=False)
          validate_dataset = WavDataset(val_fold, test=True)
          
          class_idxs = train_dataset.get_class_idxs()

          batch_sampler = SamplerFactory().get(
               class_idxs=class_idxs,
               batch_size=config.batch_size,
               n_batches=int(len(train_dataset)/config.batch_size),
               alpha=config.balance_alpha,
               kind='random'
          )

          # Data loader
          train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                                       num_workers=8, pin_memory=True)

          validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=1, shuffle=True,
                                                       num_workers=8, pin_memory=True)
          
          

          # Prepare model
          if config.model == "CRNN":
               model = CRNN().to(device)
               summary(model, (200, 64))
          elif config.model == "Wav2VecClassifier":
               model = Wav2VecClassifier().to(device)
          elif config.model == "CRNN_Chunk":
               model = CRNN_Chunk().to(device)
               summary(model, (200, 64))

               
          if LogWanDB:
               wandb.watch(model)
               
          criterion = torch.nn.MSELoss(reduction='mean')
               
          print('\nCreate model:')

          # Optimizer
          optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=False)

          best_epoch = 0; pat_cnt = 0; pat_learn_rate = 0; best_loss = 99999
          tr_loss, val_F1, val_ER = [0] * config.stop_iteration, [0] * config.stop_iteration, [0] * config.stop_iteration

          # Train on mini batches
          tr_batch_loss = list()      
          
          iter = 0
          sw = SummaryWriter(os.path.join(output_model, 'logs/'+str(fold)))
          
          for epoch in tqdm.tqdm(range(config.stop_iteration)):
               
               model.train()
               epoch_global+=1
               # TRAIN
               for (batch_data, batch_target, _) in train_loader:
                    # print(len(train_loader))
                    iter += 1
                    # Zero gradients for every batch
                    optimizer.zero_grad()
                    
                    # print(batch_data.shape)

                    batch_output, embeddings = model(move_data_to_device(batch_data, device))
                    
                    # print(batch_output.shape)
                    # print("====", batch_target.shape)

                    # Calculate loss
                    loss = criterion(batch_output, batch_target.to(device))
                         
                    tr_batch_loss.append(loss.item())

                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    if iter % config.log_iteration == 0:
                         print(batch_output[0])
                         # print(batch_target[0])
                         

               tr_loss[epoch] = np.mean(tr_batch_loss)
               
               # VALIDATE
               model.eval()

               with torch.no_grad():
                    
                    running_loss = 0.0
                    for (batch_data, batch_target, file) in validate_loader:
               
                         batch_output, embedding = model(move_data_to_device(batch_data, device))

                         loss = criterion(batch_output, batch_target.to(device))

                         running_loss += loss

                    avg_vloss = running_loss /len(validate_loader)
                    
                    # Check if during the epochs the ER does not improve
                    if avg_vloss < best_loss:
                         best_model = model
                         best_epoch = epoch
                         best_loss = avg_vloss
                         pat_cnt = 0
                         pat_learn_rate = 0

                         torch.save(best_model.state_dict(), f'{output_model}/best_fold{fold}.bin')

               pat_cnt += 1
               pat_learn_rate += 1

               if pat_learn_rate > int(0.3 * config.stop_iteration):
                    for g in optimizer.param_groups:
                         g['lr'] = g['lr']/10
                         pat_learn_rate = 0
                         print(f'\tDecreasing learning rate to:{g["lr"]}')

               print(f'Epoch: {epoch} - Train loss: {round(tr_loss[epoch],3)} - Val loss: {round(avg_vloss.item(),3)}'
                    f' - val F1 {round(val_F1[epoch]*100,2)} - val ER {round(val_ER[epoch],3)}'
                    f' - best epoch {best_epoch} F1 {round(val_F1[best_epoch]*100,2)}')

               sw.add_scalar("train_mse_loss", round(tr_loss[epoch],3), epoch)
               sw.add_scalar("validation_mse_loss", round(avg_vloss.item(),3), epoch)
               # sw.add_scalar(str(fold)+"/learning_rate", round(avg_vloss.item(),3), epoch_global)
               # Stop learning
               if (epoch == config.stop_iteration) or (pat_cnt > config.patience):
                    break
     
     if LogWanDB:
          wandb.finish()
     
if __name__ == '__main__':

     train()