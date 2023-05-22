#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, Sampler


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
    
    
def get_sampler(dataset, weighted=False, mini=False):
    
    mini_samples = 10000
    
    if weighted:
        # get the sample weights ;TODO: directly get normalized sample weights that sum to 1 
        class_counts = dataset._target_df.label.value_counts().values
        num_samples = int(sum(class_counts))
        labels = dataset._target_df.label.values
        class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
        sample_weights = np.array([class_weights[labels[i]] for i in range(int(num_samples))])

        # define the sampler
        sample_weights = torch.from_numpy(sample_weights)
        sample_weights = sample_weights.double()
        # value counts sorts by frequencies, desc order; can change this to min/len to accomodate multi-class
        nsamples = mini_samples if mini else num_samples
        sampler = CustomWeightedRandomSampler(sample_weights, nsamples, replacement=True)
    
    else:
        if mini:
            nsamples = mini_samples # 10000 samples if mini
            sampler = RandomSampler(dataset, num_samples=nsamples, replacement=True)
        else:
            sampler = None # RandomSampler(dataset, num_samples=None, replacement=False)
    
    return sampler


def make_train_samplers(dataset, args):
    dataset.set_split('train')
    # generate train sampler
    train_sampler = get_sampler(dataset, weighted=True, mini=args.pilot)
    dataset.set_split('valid')
    # generate valid sampler
    valid_sampler = get_sampler(dataset, weighted=False, mini=args.pilot)
    return train_sampler, valid_sampler


def get_test_sampler(dataset, mini=False):
    if mini:
        nsamples = int(2e5) # return 200k samples
        sampler = RandomSampler(dataset, num_samples=nsamples, replacement=True)        
    
    else:
        sampler = None    
    return sampler
