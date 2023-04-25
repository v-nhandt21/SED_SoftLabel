# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchaudio
import augment
import argparse
import random

from dataclasses import dataclass

class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)

class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max
    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)

@dataclass
class RandomReverb:
     reverberance_min: int = 50
     reverberance_max: int = 50
     damping_min: int = 50
     damping_max: int = 50
     room_scale_min: int = 0
     room_scale_max: int = 100

     def __call__(self):
          reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
          damping = np.random.randint(self.damping_min, self.damping_max + 1)
          room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)

          return [reverberance, damping, room_scale]

class SpecAugmentBand:
     def __init__(self, sampling_rate, scaler):
          self.sampling_rate = sampling_rate
          self.scaler = scaler

     @staticmethod
     def freq2mel(f):
          return 2595. * np.log10(1 + f / 700)

     @staticmethod
     def mel2freq(m):
          return ((10.**(m / 2595.) - 1) * 700)

     def __call__(self):
          F = 27.0 * self.scaler
          melfmax = freq2mel(self.sample_rate / 2)
          meldf = np.random.uniform(0, melfmax * F / 256.)
          melf0 = np.random.uniform(0, melfmax - meldf)
          low = mel2freq(melf0)
          high = mel2freq(melf0 + meldf)
          return f'{high}-{low}'


def augmentation_factory(description, sampling_rate):

     t_ms=50
     pitch_shift_max=300
     pitch_quick=True
     room_scale_min=0
     room_scale_max=100
     reverberance_min=50
     reverberance_max=50
     damping_min=50
     damping_max=50
     clip_min=0.5
     clip_max=1.0

     chain = augment.EffectChain()
     description = description.split(',')

     for effect in description:
          if effect == 'bandreject':
               chain = chain.sinc('-a', '120', SpecAugmentBand(sampling_rate, band_scaler))
          elif effect == 'pitch':
               pitch_randomizer = RandomPitchShift(pitch_shift_max)
               if pitch_quick:
                    chain = chain.pitch('-q', pitch_randomizer).rate('-q', sampling_rate)
               else:
                    chain = chain.pitch(pitch_randomizer).rate(sampling_rate)
          elif effect == 'reverb':
               randomized_params = RandomReverb(reverberance_min, reverberance_max, 
                                   damping_min, damping_max, room_scale_min, room_scale_max)
               chain = chain.reverb(randomized_params).channels()
          elif effect == 'time_drop':
               chain = chain.time_dropout(max_seconds=t_ms / 1000.0)
          elif effect == 'clip':
               chain = chain.clip(RandomClipFactor(clip_min, clip_max))
          elif effect == 'none':
               pass
          else:
               raise RuntimeError(f'Unknown augmentation type {effect}')
     return chain

def get_augment_wav(x):
     
     sampling_rate = 16000
     chain = "pitch,clip,reverb"
     augmentation_chain = augmentation_factory(chain, sampling_rate)

     y = augmentation_chain.apply(x, 
               src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
               target_info=dict(rate=sampling_rate, length=0)
     )
     aug_audio = y #.detach().numpy()[0]

     return aug_audio

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg

class SpecAugment:
    def __init__(self,num_mask=2, freq_masking=0.15, time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
            self.num_mask,
            self.freq_masking,
            self.time_masking,
            image.min())

def spec_augment(spec, num_mask=2, freq_masking=0.15, time_masking=0.20, value=0):

    spec = spec.clone()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec

class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg
       
def get_augment_mel(
                spec_num_mask=2,
                spec_freq_masking=0.15,
                spec_time_masking=0.20,
                spec_prob=0.5):

    transforms = Compose([
        UseWithProb(SpecAugment(num_mask=spec_num_mask, freq_masking=spec_freq_masking, time_masking=spec_time_masking), spec_prob),
    ])

    return transforms

if __name__ == '__main__':
     x, _ = torchaudio.load("/home/nhandt23/Desktop/DCASE/DATA/Raw/development_audio_splited/residential_area/residential_area_00_4.wav")
     print(get_augment_wav(x).shape)
     
     print(x.shape)
     mel = torch.rand(80,76)
     print(mel.size())
     trans = get_augment_mel()
     aug_spec = trans(mel)
     print(aug_spec.shape)