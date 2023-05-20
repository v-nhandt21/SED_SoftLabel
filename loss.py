import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
          

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
     
class LabelSmoothingLoss(torch.nn.Module):
     def __init__(self, smoothing=0.5):

          super(LabelSmoothingLoss, self).__init__()
          self.confidence = 1.0 - smoothing
          self.smoothing = smoothing

     def forward(self, x, target):
          target = torch.argmax(target, -1)
          logprobs = torch.nn.functional.log_softmax(x, dim=-1)
          nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
          nll_loss = nll_loss.squeeze(1)
          smooth_loss = -logprobs.mean(dim=-1)
          loss = self.confidence * nll_loss + self.smoothing * smooth_loss
          return loss.mean()
     
class GlobalLocalLoss(torch.nn.Module):
     def __init__(self):

          super(GlobalLocalLoss, self).__init__()
        #   self.queryLoss = nn.KLDivLoss(reduction='batchmean')
          self.queryLoss = nn.CrossEntropyLoss()
          self.focalLoss = FocalLoss()
          self.mse = torch.nn.MSELoss(reduction='mean')

     def forward(self, x, target):
          global_input, local_input = x 
          target, target_scene = target
          
          global_loss = self.focalLoss(global_input, target)
          
          
          local_input = None
        #   local_loss = self.queryLoss(torch.log(local_input), target_scene) #[:,0,:].squeeze(1))
          if local_input == None:
              return self.mse(global_input, target) # + global_loss
          
          local_loss = self.queryLoss(local_input, target_scene)
          loss = self.mse(global_input, target) + global_loss + local_loss
          
          return loss

if __name__ == '__main__':
     
     global_input = torch.rand(16, 5, 11)
     local_input = torch.rand(16, 11)
     
     target = torch.rand(16,5,11)
     criterion = GlobalLocalLoss()
     
     loss = criterion((global_input, local_input), target)
     print(loss)
