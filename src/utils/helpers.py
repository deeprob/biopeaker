#!/usr/bin/env python
# coding: utf-8

import os
from argparse import Namespace

import numpy as np
import pandas as pd

import torch
from .models import TFPerceptron, TFMLP, ResNet 


##############
# initialize #
##############

def get_classifier_object(classifier):
    classifier_dict = {
        "mlp": TFMLP, "linear": TFPerceptron
    }
    return classifier_dict[classifier]

def get_encoder_object(encoder):
    encoder_dict = {
        "resnet": ResNet, "homer":None, "kmer":None
        } 
    return encoder_dict[encoder]

def get_encoder_feature_size(encoder):
    encoder_feat_dict = {
        "resnet": 2200, "homer":802, "kmer":None
        }
    return encoder_feat_dict[encoder]

def get_vectorizer(encoder):
    vectorizer_dict = {
        "resnet": "ohe", "homer": "homer", "kmer":"kmer"
    }
    return vectorizer_dict[encoder]

def initialize_from_cli(cli_args):
    args = Namespace(
        # Data and Path information
        dataset=cli_args.dataset,
        genome_fasta=cli_args.genome_fasta,
        save_dir=cli_args.save_dir,
        addn_feat_dataset=cli_args.addn_feat_dataset,
        # encoder information
        encoder_name=cli_args.encoder,
        encoder=get_encoder_object(cli_args.encoder),
        encoder_state_file=f'{cli_args.encoder}.pth',
        homer_saved=cli_args.homer_saved,
        homer_pwm_motifs=cli_args.homer_pwm_motifs, 
        homer_outdir=cli_args.homer_outdir,
        k=cli_args.kmer,
        feat_size=get_encoder_feature_size(cli_args.encoder),
        addn_feat_size=cli_args.addn_feat_size,
        vectorizer=get_vectorizer(cli_args.encoder),
        # classifier information
        classifier_name=cli_args.classifier,
        classifier=get_classifier_object(cli_args.classifier),
        classifier_state_file=f'{cli_args.classifier}.pth',
        dropout_prob=cli_args.dropout_prob,
        # Training hyper parameters
        batch_size=cli_args.batch_size,
        early_stopping_function=cli_args.early_stopping_function,
        early_stopping_criteria=cli_args.early_stopping_criteria,
        learning_rate=cli_args.learning_rate,
        num_epochs=cli_args.num_epochs,
        tolerance=cli_args.tolerance,
        seed=cli_args.random_seed,
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True if cli_args.pytorch_device=="cuda" else False,
        expand_filepaths_to_save_dir=True,
        pilot=cli_args.pilot,
        train=not cli_args.test,
        train_encoder=not cli_args.freeze_encoder,
        test_batch_size=cli_args.test_batch_size,
        integrated_gradients=cli_args.integrated_gradients,
    )

    if not torch.cuda.is_available():
        args.cuda = False

    if args.expand_filepaths_to_save_dir:
        args.encoder_state_file = os.path.join(args.save_dir, args.encoder_state_file)
        args.classifier_state_file = os.path.join(args.save_dir, args.classifier_state_file)
    
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return


####################
# dataset creation #
####################

def read_bed_to_df(bed_file):
    df = pd.read_csv(bed_file, sep="\t", header=None, usecols=[0,1,2])
    df.columns = ["chrm", "start", "end"]
    return df

def create_tf_dataset_file(peak_bed_path, non_peak_bed_path, split_ratio=(70,20,10)):
    peak_df = read_bed_to_df(peak_bed_path)
    non_peak_df = read_bed_to_df(non_peak_bed_path)
    peak_df["label"] = 1
    non_peak_df["label"] = 0
    df = pd.concat((peak_df, non_peak_df),  axis=0).reset_index(drop=True)
    assert sum(split_ratio) == 100
    df["split"] = "train"
    a = np.arange(len(df))
    np.random.shuffle(a)
    for arr_idx, split_val in zip(np.split(a, [int(split_ratio[0]/100 * len(a)), int((split_ratio[0]/100 + split_ratio[1]/100) * len(a))]), ["train", "valid", "test"]):
        df.loc[arr_idx, "split"] = split_val
    return df
