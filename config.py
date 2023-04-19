import numpy as np

mel_input=True
sample_rate=16000
hop_size=128
n_fft=2048
n_mel=64

data_dir = "/home/nhandt23/Desktop/DCASE/DATA/Raw/"
n_classes = 11
freeze_encoder = False

learning_rate=0.0005
batch_size=128
stop_iteration = 20
patience = int(0.9*stop_iteration)
log_iteration=500

holdout_fold=np.arange(1, 6)

balance_alpha=0.5
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