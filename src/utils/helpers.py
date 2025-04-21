#!/usr/bin/env python
# coding: utf-8

import os
from argparse import Namespace

import numpy as np
import pandas as pd
import pybedtools

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
        "resnet": 4200, "homer":802, "kmer":None
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


###################
# bedfile parsing #
###################

# create peaks centered around the given peak bed file 
# with an user provided input that cannot be less than 500 bp
# do not extend beyond chrom sizes

def create_centered_peaks(bed_file, window_size, chrom_sizes):
    """
    Create peaks centered around given peak coordinates with a user-defined window size.
    Ensures peaks remain exactly `window_size` bp long, adjusting within chromosome limits.
    Parameters:
    - bed_file (str): Path to the narrowPeak BED file.
    - window_size (int): Desired window size (minimum 500 bp).
    - chrom_sizes (dict): Dictionary of chromosome sizes.
    Returns:
    - DataFrame with fixed-size centered peaks.
    """
    if window_size < 500:
        raise ValueError("Window size cannot be less than 500 bp.")
    # Load the BED file
    cols = ["chrom", "start", "end"]
    df = pd.read_csv(bed_file, sep="\t", header=None, names=cols, usecols=[0, 1, 2])
    # Filter out invalid chromosomes early
    df = df[df["chrom"].isin(chrom_sizes)]
    # Determine offset from center
    df_peaks = pd.DataFrame()
    for position, offset in zip(["center", "left", "right"], [window_size//2, window_size//4, 3 * window_size//4]):
        # Compute initial start and end
        peak_center = (df["start"] + df["end"]) // 2
        new_start = peak_center - offset 
        new_end = new_start + window_size
        # Get chromosome max size
        chrom_max_sizes = df["chrom"].map(chrom_sizes)
        # Shift if out of bounds
        shift_right = np.where(new_start < 1, -new_start+1, 0)  # Extra bp to shift right
        shift_left = np.where(new_end > chrom_max_sizes, new_end - chrom_max_sizes, 0)  # Extra bp to shift left
        # Adjust start and end while keeping size fixed
        new_start = new_start + shift_right - shift_left
        new_end = new_end + shift_right - shift_left
        assert np.all(new_end-new_start==window_size)
        # Create final dataframe
        df_centered = pd.DataFrame({
            "chrom": df["chrom"],
            "start": new_start,
            "end": new_end
        })
        df_peaks = pd.concat([df_peaks, df_centered])
    return df_peaks

def generate_non_overlapping_windows(chrom_sizes, window_size=1000):
    """
    Generate non-overlapping genomic windows of fixed size for given chromosome sizes.
    Args:
    - chrom_sizes (dict): Dictionary mapping chromosome names to their lengths.
    - window_size (int): Length of each window (default=1000).
    Returns:
    - pd.DataFrame: DataFrame containing columns ["chrom", "start", "end"]
    """
    windows = []
    for chrom, size in chrom_sizes.items():
        # Generate start positions at intervals of `window_size`
        starts = np.arange(1, size - window_size + 1, window_size)
        ends = starts + window_size  # Compute end positions
        # If last end exceeds chromosome size, shift the last window
        if ends[-1] < size:
            last_end = size
            last_start = size - window_size
            starts = np.append(starts, [last_start])
            ends = np.append(ends, [last_end])
        # Store results as tuples (chrom, start, end)
        windows.extend(zip([chrom] * len(starts), starts, ends))
    return pd.DataFrame(windows, columns=["chrom", "start", "end"])


def generate_peak_files(peakfile_path, window_size, chrom_sizes, save_dir, overlap_frac=0.5):
    norm_bed_df = create_centered_peaks(peakfile_path, window_size, chrom_sizes)
    genome_df = generate_non_overlapping_windows(chrom_sizes, window_size)
    pybedtools.helpers.set_tempdir(save_dir)
    genome_bed = pybedtools.BedTool.from_dataframe(genome_df).sort()
    peak_bed = pybedtools.BedTool.from_dataframe(norm_bed_df).sort()
    non_peak_bed = genome_bed.intersect(peak_bed, v=True, F=overlap_frac)
    peak_path = peak_bed.saveas(os.path.join(save_dir, "peaks.bed.gz"))
    non_peak_path = non_peak_bed.saveas(os.path.join(save_dir, "not_peaks.bed.gz"))
    pybedtools.helpers.cleanup(verbose=False, remove_all=True)
    return peak_path, non_peak_path
