from transformers import Wav2Vec2Model
import torch
import torch.nn as nn
import config

class Wav2VecClassifier(nn.Module):
     
     def __init__(self):     
          super(Wav2VecClassifier, self).__init__()
          self.backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', gradient_checkpointing = False)
          self.backbone.config.mask_time_prob = 0.3
          self.backbone.config.mask_time_min_masks = 5
          self.backbone.config.mask_feature_prob = 0.3
          self.backbone.config.mask_feature_min_masks = 2

          if int(config.freeze_encoder) == 1:
               self.backbone.feature_extractor._freeze_parameters()
          else:
               print("Do not freeze")

          self.bottleneck = nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.25),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128))
          self.classifier = nn.Linear(128, config.n_classes)
          self.sigmoid = nn.Sigmoid()
     
     def forward(self, x):
          x = self.backbone(x, output_hidden_states = True, return_dict = True)
          x = torch.mean(x.last_hidden_state, dim=1)

          embedding = self.bottleneck(x)
          x = self.classifier(embedding)
          # x = torch.nn.functional.softmax(x, dim=1)
          x = self.sigmoid(x)
          return x, embedding
     
class CRNN(nn.Module):
     def __init__(self, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2):
          super(CRNN, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
          self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
          self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
          
          self.dropout = nn.Dropout(_dropout_rate)

          self.gru1 = nn.GRU(int(3*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

          self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)


          self.linear2 = nn.Linear(rnn_hid, config.n_classes)
          
          self.sigmoid = nn.Sigmoid()

     def forward(self, input):

          x = self.conv1(input[:,None,:,:])

          x = self.batch_norm1(x)
          x = torch.relu(x)
          x = self.pool1(x)
          x = self.dropout(x)
          
          x = self.conv2(x)
          x = self.batch_norm2(x)
          x = torch.relu(x)
          x = self.pool2(x)
          x = self.dropout(x)

          x = self.conv3(x)
          x = self.batch_norm3(x)
          x = torch.relu(x)
          x = self.pool3(x)
          x = self.dropout(x)

          x = x.permute(0, 2, 1, 3)
          x = x.reshape((x.shape[0], x.shape[1], -1))
          
          # Bidirectional layer
          recurrent, _ = self.gru1(x)
          x = self.linear1(recurrent)
          # x = torch.mean(x, dim=1)
          embedding = x.squeeze(dim=1)          
          x = self.linear2(embedding)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return x, embedding
     
class CRNN_Chunk(nn.Module):
     def __init__(self, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2):
          super(CRNN_Chunk, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
          self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
          
          self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
          self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
          self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
          
          self.dropout = nn.Dropout(_dropout_rate)

          self.gru1 = nn.GRU(int(3*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

          self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)


          self.linear2 = nn.Linear(rnn_hid, config.n_classes)
          
          self.sigmoid = nn.Sigmoid()

     def forward(self, input):

          x = self.conv1(input[:,None,:,:])

          x = self.batch_norm1(x)
          x = torch.relu(x)
          x = self.pool1(x)
          x = self.dropout(x)
          
          # print(x.size())
          
          x = self.conv2(x)
          x = self.batch_norm2(x)
          x = torch.relu(x)
          x = self.pool2(x)
          x = self.dropout(x)
          

          x = self.conv3(x)
          x = self.batch_norm3(x)
          x = torch.relu(x)
          x = self.pool3(x)
          x = self.dropout(x)

          x = x.permute(0, 2, 1, 3)
          x = x.reshape((x.shape[0], x.shape[1], -1))
          
          # Bidirectional layer
          recurrent, _ = self.gru1(x)
          x = self.linear1(recurrent)
          # x = torch.mean(x, dim=1)
          embedding = x.squeeze(dim=1)          
          x = self.linear2(embedding)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return x, embedding