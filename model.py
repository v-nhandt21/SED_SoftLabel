from transformers import Wav2Vec2Model
import torch
import torch.nn as nn
import config

class CondionalSpeakerLayerNorm(nn.Module):
     """ Speaker Condition Layer Normalization: https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer/blob/341173fdb44d6fadb4d8f68f83f0bce9c0072ddc/model/blocks.py#L8 """

     def __init__(self, speaker_embed_size, x_dim):
          super(CondionalSpeakerLayerNorm, self).__init__()

          self.speaker_embed_size = speaker_embed_size
          self.x_dim = x_dim
          
          self.speaker_linear = torch.nn.Linear( self.speaker_embed_size, 2 * self.x_dim, bias=False) # For both b (bias) and g (gain)
          
          torch.nn.init.xavier_uniform_(self.speaker_linear.weight)
          
          self.eps = 1e-8

     def forward(self, x, spker_embed):

          # Normalize Input Features
          mean = torch.mean(x, dim=-1, keepdim=True)
          sigma = torch.std(x, dim=-1, keepdim=True)
          y = (x - mean) / (sigma + self.eps)  # [B, T, H_m]
          # Get Bias and Gain
          # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]
          bias, scale = torch.split(self.speaker_linear(spker_embed), self.x_dim, dim=-1)
          # Perform Scailing and Shifting
          y = scale * y + bias  # [B, T, H_m]
          return y

class Wav2VecClassifier(nn.Module):
     
     def __init__(self):     
          super(Wav2VecClassifier, self).__init__()
          self.backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', gradient_checkpointing = False)
          self.backbone.config.mask_time_prob = 0.3
          self.backbone.config.mask_time_min_masks = 5
          self.backbone.config.mask_feature_prob = 0.3
          self.backbone.config.mask_feature_min_masks = 2

          if int(config.freeze_encoder) == 1:
               print("Freeze extractor")
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
          self.classifier = nn.Linear(128, config.n_scene_classes)
          self.emb = nn.Linear(128, 32)
          self.sigmoid = nn.Sigmoid()
     
     def forward(self, x):
          x = self.backbone(x, output_hidden_states = True, return_dict = True)
          x = torch.mean(x.last_hidden_state, dim=1)

          embedding_128 = self.bottleneck(x)
          embedding = self.emb(embedding_128)        
          x = self.classifier(embedding_128)
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
          x = torch.mean(x, dim=1)
          embedding = x.squeeze(dim=1)          
          x = self.linear2(embedding)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return x, embedding
     

class CRNN_Chunk_BK(nn.Module):
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
          
          if config.attention:
               
               self.prob = nn.Softmax(dim=1)
               self.frame_sec = int(config.sample_rate/config.hop_size)
               self.saln = CondionalSpeakerLayerNorm(32, config.chunk_size*self.frame_sec)

     def forward(self, input):
          batch = input.size(0)
          seq_lenght = input.size(1)

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
          
          
          
          
          ###############
          if config.attention:
               feature1 = embedding[:,int(config.chunk_size/2)*self.frame_sec:(int(config.chunk_size/2)+1)*self.frame_sec,:] # batch, 5,32
               embedding_query = torch.mean(feature1, dim=1).squeeze(dim=1)  # batch, 32
               attention_score = embedding * embedding_query.unsqueeze(1)
               attention_score = torch.sum(attention_score, dim=-1)
               
               attention_distribution = self.prob(attention_score)                                       # [batch, seq-len]
               attention_output = torch.bmm(
                    embedding.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
                    attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
               )#[batch, hidden-node,1]
               
               x = self.saln(embedding.permute(0,2,1), attention_output.permute(0,2,1)).permute(0,2,1)
          ###########################
          
          
          x = self.linear2(embedding)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return x, embedding
     
class CRNN_GLA(nn.Module):
     def __init__(self, cnn_filters=128, rnn_hid=64, _dropout_rate=0.2):
          super(CRNN_GLA, self).__init__()
          self.global_CRNN = CRNN_Chunk()
          
          if config.query_model == "W2V":
               self.query_model = Wav2VecClassifier()
          else:
               self.query_model = CRNN()
          
          self.linear2 = nn.Linear(rnn_hid, 32)
          self.linear3 = nn.Linear(32, config.n_classes)
          
          self.linear_scene = nn.Linear(32, config.n_scene_classes)
          
          self.sigmoid = nn.Sigmoid()
          self.prob = nn.Softmax(dim=1)
          
          self.frame_sec = int(config.sample_rate/config.hop_size)
          self.transform = nn.Linear(1, self.frame_sec*config.chunk_size)

     def forward(self, input):
          
          global_input, local_input = input
          
          batch = global_input.size(0)
          seq_lenght = global_input.size(1)

          x1, embedding_global = self.global_CRNN(global_input) # emb1: torch.Size([16, 205, 32])
          
          feature1 = embedding_global[:,int(config.chunk_size/2)*self.frame_sec:(int(config.chunk_size/2)+1)*self.frame_sec,:] # batch, 5,32
          
          ############
          
          embedding_query = torch.mean(feature1, dim=1).squeeze(dim=1)  # batch, 32
          
          
          attention_score = embedding_global * embedding_query.unsqueeze(1)
          attention_score = torch.sum(attention_score, dim=-1)
          
          attention_distribution = self.prob(attention_score)                                       # [batch, seq-len]
          
          attention_output = torch.bmm(
               embedding_global.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
               attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
          )                                                                                #[batch, hidden-node,1]
          
          query_feature = self.linear_scene(attention_output.squeeze(dim=2)) 
          
          attention_output = self.transform(attention_output)   #[batch, 32,1] -> [batch, 32, 205]
          feature2 = attention_output.permute(0, 2, 1)
          embedding = torch.cat([embedding_global, feature2],dim=2)
          
          x = self.linear2(embedding)
          x = self.linear3(x)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return (x, query_feature), embedding
     
     
          ############################################
     
          
          query_feature, embedding_query = self.query_model(local_input) # emb2: torch.Size([16, 32])
          
          attention_score = embedding_global * embedding_query.unsqueeze(1)
          attention_score = torch.sum(attention_score, dim=-1)
          
          attention_distribution = self.prob(attention_score)                                       # [batch, seq-len]
          
          attention_output = torch.bmm(
               embedding_global.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
               attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
          )                                                                                #[batch, hidden-node,1]
          
          attention_output = self.transform(attention_output)
          feature2 = attention_output.permute(0, 2, 1)
          
          
          embedding = torch.cat([feature1, feature2],dim=2)
          
          x = self.linear2(embedding)
          x = self.linear3(x)
          
          if config.use_sigmoid:
               x = self.sigmoid(x)
          # x = torch.nn.functional.softmax(x, dim=1)

          return (x, query_feature), embedding
     
if __name__ == '__main__':
     
     # global_input = torch.Tensor(16, 205, 64)
     # local_input = torch.Tensor(16, 205, 64)
     
     # model = CRNN_GLA()
     
     # out, emb = model((global_input, local_input ))
     
     # # embedding = model(input)
     # print(out.shape)
     # print(emb.shape)
     
     
     ############
     x = torch.Tensor(16, 16000)
     model =Wav2VecClassifier()
     out, emb = model(x)
     print(out.shape)
     print(emb.shape)