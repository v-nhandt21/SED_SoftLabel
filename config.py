import numpy as np

##### Model
model = "CRNN_Chunk"
n_classes = 11
freeze_encoder = True
use_sigmoid=True

##### Training strategy
learning_rate=0.001
batch_size=16
stop_iteration = 30
patience = int(0.9*stop_iteration)
log_iteration=500
holdout_fold=np.arange(1, 6)
chunk_size=41

##### Data Augment
data_dir = "/home/nhandt23/Desktop/DCASE/DATA/Raw/"
balance_alpha=0.5
wavaugment=True
specaugment=True

sample_rate=16000
hop_size=3200
win_size=hop_size*2
n_fft=6400
n_mel=64

#################################### 
# For the hard labels we have 11 classes
labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 'wind_blowing',
          'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 'metro approaching', 'metro leaving']
class_labels_hard = {
     'birds_singing': 0,
     'car': 1,
     'people talking': 2,
     'footsteps': 3,
     'children voices': 4,
     'wind_blowing': 5,
     'brakes_squeaking': 6,
     'large_vehicle': 7,
     'cutlery and dishes': 8,
     'metro approaching': 9,
     'metro leaving': 10,
}
posterior_thresh = 0.5
