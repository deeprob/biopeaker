import os
import gzip

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, DeepLiftShap

import tqdm

from .samplers import get_test_sampler, generate_batches
from .datasets import load_data


def save_linear_model_features(classifier, vectorizer, homer_saved, save_file):
    # get weights
    for layer in classifier.children():
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
    
    def forward(self, x_in, a_in=None):
        if self.encoder:
            x_in = self.encoder(x_in)
        y_out = self.classifier(x_in, a_in)
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

def permute_sequence(seq):
    batch_size = seq.shape[0]
    subset_batch_size = batch_size//4
    subset_seq = seq[torch.randperm(batch_size)[:subset_batch_size]]
    permuted_seq = torch.zeros_like(subset_seq, device=seq.device)
    for i in range(subset_batch_size):
        seq_permutation = torch.randperm(seq.shape[-1])
        permuted_seq[i] = subset_seq[i, :, seq_permutation]
    return permuted_seq

def baseline_func(x):
    if isinstance(x, tuple):
        if x[1].shape[1]>0:
            print("got addn features")
            permuted_seq = permute_sequence(x[0])
            return (permuted_seq, torch.zeros(x[1].shape[0]//4, x[1].shape[-1], device=x[1].device))
    permuted_seq = permute_sequence(x)
    return permuted_seq

def eval_model(args, dataset_split="test"):
    """
    classifier initialized before
    dataset of type TFDataset
    """

    # Loading the dataset
    dataset = load_data(args.dataset, args.genome_fasta, args.vectorizer, addn_feat_path=args.addn_feat_dataset, k=args.k, homer_saved=args.homer_saved, homer_pwm_motifs=args.homer_pwm_motifs, homer_outdir=args.homer_outdir)
    
    # Initializing encoder
    if args.encoder:
        encoder = args.encoder()
        encoder.load_state_dict(torch.load(args.encoder_state_file))
        encoder.eval()
    else:
        encoder = None
    
    # Initializing classifier
    classifier = args.classifier(args)
    classifier.load_state_dict(torch.load(args.classifier_state_file))
    classifier.eval()

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

    if args.interpreter:
        interpreter = args.interpreter(model)
        seq_feat_array = None
        seq_attr_array = None
        addn_feat_array = None
        addn_attr_array = None
        genomic_loc_array = None
    
    # Runnning evaluation routine
    test_bar = tqdm.tqdm(desc=f'split={dataset_split}',
                          total=len(dataset)//args.test_batch_size, 
                          position=0, 
                          leave=True)

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        seq_feats = batch_dict['x_data'].float()
        add_feats = None
        if args.addn_feat_size>0:
            add_feats = batch_dict['a_data'].float()
        y_pred = model(x_in=seq_feats, a_in=add_feats)
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

        # model interpretation with integrated gradients/ deep shap
        if args.interpreter:
            interpreter_kwargs = {}
            if isinstance(interpreter, IntegratedGradients):
                interpreter_kwargs["internal_batch_size"] = args.test_batch_size
                interpreter_kwargs["n_steps"] = 500
            
            elif isinstance(interpreter, DeepLiftShap):
                interpreter_kwargs["baselines"] = baseline_func

            if args.addn_feat_size>0:
                (seq_attr, addn_attr), approximation_error = interpreter.attribute((seq_feats, add_feats), return_convergence_delta=True, **interpreter_kwargs)
                addn_attr = addn_attr.cpu().detach().numpy()
                add_feats = add_feats.cpu().detach().numpy()
            else:
                seq_attr, approximation_error = interpreter.attribute(seq_feats, return_convergence_delta=True, **interpreter_kwargs)
            
            seq_attr = seq_attr.cpu().detach().numpy()
            seq_feats = seq_feats.cpu().detach().numpy()
            if seq_attr_array is None:
                seq_attr_array = seq_attr
                seq_feat_array = seq_feats
                genomic_loc = batch_dict["genome_loc"]
                genomic_loc = np.concatenate((np.array(genomic_loc[0]).reshape(-1,1), genomic_loc[1].cpu().numpy().reshape(-1,1), genomic_loc[2].cpu().numpy().reshape(-1,1)), axis=1)
                genomic_loc_array = genomic_loc
                if args.addn_feat_size>0:
                    addn_attr_array = addn_attr
                    addn_feat_array = add_feats
            else:
                seq_attr_array = np.concatenate((seq_attr_array, seq_attr), axis=0)
                seq_feat_array = np.concatenate((seq_feat_array, seq_feats), axis=0)
                genomic_loc = batch_dict["genome_loc"]
                genomic_loc = np.concatenate((np.array(genomic_loc[0]).reshape(-1,1), genomic_loc[1].cpu().numpy().reshape(-1,1), genomic_loc[2].cpu().numpy().reshape(-1,1)), axis=1)
                genomic_loc_array = np.concatenate((genomic_loc_array, genomic_loc), axis=0)
                if args.addn_feat_size>0:
                    addn_attr_array = np.concatenate((addn_attr_array, addn_attr), axis=0)
                    addn_feat_array = np.concatenate((addn_feat_array, add_feats), axis=0)
        
        # update test bar
        test_bar.set_postfix(loss=running_loss, 
                              batch=batch_index)
        test_bar.update()
    
    # interpret model features at the end for linear classifier with homer features
    if args.classifier_name == "linear":
        if args.encoder_name == "homer":
            save_file = os.path.join(args.save_dir, "features.csv")
            save_linear_model_features(classifier, args.vectorizer, args.homer_saved, save_file)

    if args.interpreter:
        save_loc_file = os.path.join(args.save_dir, "locations.npy")
        save_seq_attr_file = os.path.join(args.save_dir, "seq_attr.npy")
        save_seq_feat_file = os.path.join(args.save_dir, "seq_feat.npy")
        np.save(save_loc_file, genomic_loc_array)
        np.save(save_seq_attr_file, seq_attr_array)
        np.save(save_seq_feat_file, seq_feat_array)
        if args.addn_feat_size>0:
            save_addn_attr_file = os.path.join(args.save_dir, "addn_attr.npy")
            save_addn_feat_file = os.path.join(args.save_dir, "addn_feat.npy")
            np.save(save_addn_attr_file, addn_attr_array)
            np.save(save_addn_feat_file, addn_feat_array)
    return save_file
