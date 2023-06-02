import os
import gzip

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

import tqdm

from .samplers import get_test_sampler, generate_batches
from .datasets import load_data


def save_linear_model_features(model, vectorizer, homer_saved, save_file):
    # get weights
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weights = layer.state_dict()['weight']
    # from vectorizer, get the feature names
    if vectorizer=="homer":
        df = pd.read_csv(homer_saved, nrows=0, index_col=0)
        features = np.array(df.columns)
        weights = weights.detach().cpu().numpy().flatten()
        assert len(features) == len(weights)
        feat_df = pd.DataFrame({"features": features, "weights": weights})
        feat_df.sort_values("weights", ascending=False).to_csv(save_file, index=False)
    else:
        raise ValueError(f"Model interpretation not available for linear model with {vectorizer} vectorizer")
    return

#################
# dl evaluation #
#################

class Model(nn.Module):
    def __init__(self, encoder, classifier):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x_in):
        if self.encoder:
            x_in = self.encoder(x_in)
        y_out = self.classifier(x_in)
        y_out = torch.sigmoid(torch.flatten(y_out))
        return y_out
    

def save_test_pred(filename, y_preds, y_targets, genomic_locs, mode="ab"):
    y_preds = y_preds.cpu().detach().numpy()
    y_targets = y_targets.cpu().detach().numpy()
    genomic_locs = list(map(lambda x: x.numpy() if type(x)==torch.Tensor else x, genomic_locs))
    
    with gzip.open(filename, mode) as f:
        for y_pred, y_target, chrm, start, end in zip(y_preds, y_targets, genomic_locs[0], genomic_locs[1], genomic_locs[2]):
            f.write(bytes(f"{y_pred},{y_target},{chrm},{start},{end}\n", "utf-8"))
    return


def eval_model(args, dataset_split="test"):
    """
    classifier initialized before
    dataset of type TFDataset
    """

    # Loading the dataset
    dataset = load_data(args.dataset, args.genome_fasta, args.vectorizer, k=args.k, homer_saved=args.homer_saved, homer_pwm_motifs=args.homer_pwm_motifs, homer_outdir=args.homer_outdir)
    
    # Initializing encoder
    encoder = args.encoder()
    if encoder:
        encoder.load_state_dict(torch.load(args.encoder_state_file))
    
    # Initializing classifier
    classifier = args.classifier(args)
    classifier.load_state_dict(torch.load(args.classifier_state_file))

    # get combined model
    model = Model(encoder, classifier)
    model = model.to(args.device)

    # Defining loss function
    loss_func = nn.BCEWithLogitsLoss()

    # Making samplers
    dataset.set_split(dataset_split)
    test_sampler = get_test_sampler(dataset, mini=args.pilot)

    batch_generator = generate_batches(dataset, sampler=test_sampler, shuffle=False, 
                                       batch_size=args.test_batch_size, 
                                       device=args.device, drop_last=False)

    ##### Evaluation Routine #####
    running_loss = 0.
    model.eval()
    mode = "wb"
    save_filename = f"{args.encoder_name}_{args.classifier_name}.csv.gz"
    save_file = os.path.join(args.save_dir, save_filename)

    if args.integrated_gradients:
        integrated_gradients = IntegratedGradients(model)
        attribution_array = None
        genomic_loc_array = None
    
    # Runnning evaluation routine
    test_bar = tqdm.tqdm(desc=f'split={dataset_split}',
                          total=len(dataset)//args.test_batch_size, 
                          position=0, 
                          leave=True)

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = model(x_in=batch_dict['x_data'].float())
        save_test_pred(save_file, 
                       torch.flatten(y_pred), 
                       batch_dict['y_target'], 
                       batch_dict["genome_loc"], 
                       mode=mode)
        mode = "ab" 

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # model interpretation with integrated gradients
        if args.integrated_gradients:
            attributions, approximation_error = integrated_gradients.attribute(batch_dict['x_data'].float(), target=0, internal_batch_size=args.test_batch_size, return_convergence_delta=True, n_steps=500)
            attributions = torch.sum(attributions, 1).cpu().numpy()
            if attribution_array is None:
                attribution_array = attributions
                genomic_loc = batch_dict["genome_loc"]
                genomic_loc = np.concatenate((np.array(genomic_loc[0]).reshape(-1,1), genomic_loc[1].cpu().numpy().reshape(-1,1), genomic_loc[2].cpu().numpy().reshape(-1,1)), axis=1)
                genomic_loc_array = genomic_loc
            else:
                attribution_array = np.concatenate((attribution_array, attributions), axis=0)
                genomic_loc = batch_dict["genome_loc"]
                genomic_loc = np.concatenate((np.array(genomic_loc[0]).reshape(-1,1), genomic_loc[1].cpu().numpy().reshape(-1,1), genomic_loc[2].cpu().numpy().reshape(-1,1)), axis=1)
                genomic_loc_array = np.concatenate((genomic_loc_array, genomic_loc), axis=0)
        
        # update test bar
        test_bar.set_postfix(loss=running_loss, 
                              batch=batch_index)
        test_bar.update()
    
    # interpret model features at the end for linear classifier with homer features
    if args.classifier_name == "linear":
        if args.encoder_name == "homer":
            save_file = os.path.join(args.save_dir, "features.csv")
            save_linear_model_features(model, args.vectorizer, args.homer_saved, save_file)

    if args.integrated_gradients:
        save_loc_file = os.path.join(args.save_dir, "locations.npy")
        save_attr_file = os.path.join(args.save_dir, "attributions.npy")
        np.save(save_loc_file, genomic_loc_array)
        np.save(save_attr_file, attribution_array)
    return save_file
