#!/usr/bin/env python
# coding: utf-8

import os
import gzip
import logging
from argparse import Namespace
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm
from sklearn.metrics import average_precision_score

from .samplers import make_train_samplers, get_test_sampler
from .models import TFPerceptron, TFMLP, ResNet 
from .datasets import load_data


#############
# arg parse #
#############

def get_classifier(model_name):
    if model_name=="resnet":
        model=ResNet
    elif model_name=="mlp":
        model=TFMLP
    else:
        model=TFPerceptron
    return model


def initialize_from_cli(cli_args):
    args = Namespace(
        # Data and Path information
        dataset=cli_args.dataset,
        vectorizer=cli_args.vectorizer,
        genome_fasta=cli_args.genome_fasta,
        homer_saved=cli_args.homer_saved,
        homer_pwm_motifs=cli_args.homer_pwm_motifs, 
        homer_outdir=cli_args.homer_outdir,
        k=cli_args.kmer,
        model_state_file=f'{cli_args.model_name}.pth',
        save_dir=cli_args.save_dir,
        feat_size=(4, 500) if cli_args.model_name=="resnet" else (1, 802),
        
        # Model hyper parameters
        model_name=cli_args.model_name,
        classifier=get_classifier(cli_args.model_name),
        dropout_prob=cli_args.dropout_prob,
        
        # Training hyper parameters
        batch_size=cli_args.batch_size,
        early_stopping_criteria=cli_args.early_stopping_criteria,
        learning_rate=cli_args.learning_rate,
        num_epochs=cli_args.num_epochs,
        tolerance=cli_args.tolerance,
        seed=cli_args.random_seed,
        
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True if cli_args.pytorch_device=="cuda" else False,
        expand_filepaths_to_save_dir=True,
        pilot=cli_args.pilot, # 2% of original dataset
        train=not cli_args.test,
        test_batch_size=cli_args.test_batch_size
    )

    if not torch.cuda.is_available():
        args.cuda = False

    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
    
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


###############
# dl training #
###############

def generate_batches(dataset, sampler, batch_size, shuffle=False,
                    drop_last=True, device="cpu"):
    
    # define the dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                           sampler=sampler, shuffle=shuffle,
                           drop_last=drop_last, num_workers=0,
                           pin_memory=False)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items(): # TODO: Look at datadict.keys(), no need for tensor here
            if name!= "genome_loc":
                out_data_dict[name] = data_dict[name].to(device)
            else:
                out_data_dict[name] = data_dict[name]
        yield out_data_dict

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 0,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_aps': [],
            'val_loss': [],
            'val_aps': [],
            'test_loss': -1,
            'test_aps': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates. Determines whether to stop model training early

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        curr_aps = train_state['val_aps'][0]
        train_state['early_stopping_best_val'] = curr_aps
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        apc_tm1, apc_t = train_state['val_aps'][-2:] # looking at the last two validation aps

        # If apc worsened 
        if apc_t <= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1 # updating early stopping info
        # apc increased
        else:
            # Save the best model
            if apc_t > train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = apc_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def train_model(args):
    # Logger config
    logging.basicConfig(filename=os.path.join(args.save_dir, f"{args.model_name}.log"), filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',)
    # Load the dataset
    logging.debug(f'Loading dataset and creating vectorizer...')
    dataset = load_data(args.dataset, args.genome_fasta, args.vectorizer, k=args.k, homer_saved=args.homer_saved, homer_pwm_motifs=args.homer_pwm_motifs, homer_outdir=args.homer_outdir)    
    logging.debug(f'Dataset loaded.')

    # Initializing model
    logging.debug(f'Initializing model...')
    classifier = args.classifier(args)

    if os.path.exists(args.model_state_file):
        logging.debug(f'Loading previous model found on path ...')
        classifier.load_state_dict(torch.load(args.model_state_file))

    classifier = classifier.to(args.device)
    logging.debug(f'Model initialized on {args.device}.')

    model_params = get_n_params(classifier)
    logging.debug(f"The model has {model_params} parameters.")
        
    # Defining loss function, optimizer and scheduler
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, eps=1e-7)
    # adjusting the learning rate for better performance
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)    
    logging.debug(f"Model learning rate set to {args.learning_rate}.")

    # Making samplers
    train_sampler, valid_sampler = make_train_samplers(dataset, args)
    logging.debug(f"Training {model_params} parameters with {train_sampler.num_samples} instances at a rate of {round(train_sampler.num_samples/model_params, 6)} instances per parameter.")
    logging.debug(f"Model batch size set to {args.batch_size}.")

    # Defining initial train state
    train_state = make_train_state(args)
    
    ##### Training Routine #####
    
    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset
            logging.debug(f"Model Training --- Epoch {epoch_index}.")

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, sampler=train_sampler,
                                               batch_size=args.batch_size, 
                                               device=args.device)
            running_loss = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):

                # the training routine as follows:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'].view(-1, 1).float())

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the loss for update
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

            train_state['train_loss'].append(running_loss)

            # Iterate over val dataset
            logging.debug(f"Model Evaluation --- Epoch {epoch_index}.")

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('valid')
            batch_generator = generate_batches(dataset, sampler=valid_sampler,
                                               batch_size=int(args.test_batch_size), 
                                               device=args.device)
            running_loss = 0.
            tmp_filename = os.path.join(args.save_dir, f"validation_file.tmp")
            tmp_file = open(tmp_filename, "wb")
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred = classifier(x_in=batch_dict['x_data'].float())
                y_target = batch_dict['y_target'].view(-1, 1).float()

                # step 3. compute the loss
                loss = loss_func(y_pred, y_target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # save data for computing aps
                for yp, yt in zip(torch.sigmoid(torch.flatten(y_pred)).cpu().detach().numpy(), torch.flatten(y_target).cpu().detach().numpy()):
                    tmp_file.write(bytes(f"{yp},{yt}\n", "utf-8"))

            train_state['val_loss'].append(running_loss)
            
            # compute aps from saved file
            tmp_file.close()
            val_aps = compute_aps_from_file(tmp_filename)
            os.remove(tmp_filename)
        
            train_state['val_aps'].append(val_aps)
            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])
            
            logging.debug(f"Epoch: {epoch_index}, Validation Loss: {running_loss}, Validation APS: {val_aps}")

            if train_state['stop_early']:
                logging.debug("Early stopping criterion fulfilled!")
                break

    except KeyboardInterrupt:
        logging.warning("Exiting loop")
    
    return train_state

def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def compute_aps(y_pred, y_target):
    """Computes the average precision score"""
    y_target = y_target.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return average_precision_score(y_target, y_pred)

def compute_aps_from_file(file):
    """Computes the aps score from a file"""
    results_df = pd.read_csv(file, header=None)
    y_target = results_df[1].values
    y_pred = results_df[0].values
    return average_precision_score(y_target, y_pred)

def get_split_ratio(bd):
    values = bd["y_target"].cpu().numpy()
    counts = Counter(values)
    return counts[1]/counts[0]

def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

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

    classifier = args.classifier(args)
    dataset = load_data(args.dataset, args.genome_fasta, args.vectorizer, k=args.k, homer_saved=args.homer_saved, homer_pwm_motifs=args.homer_pwm_motifs, homer_outdir=args.homer_outdir)
    
    # Initializing
    classifier.load_state_dict(torch.load(args.model_state_file))
    classifier = classifier.to(args.device)
    loss_func = nn.BCEWithLogitsLoss()

    dataset.set_split(dataset_split)
    
    test_sampler = get_test_sampler(dataset, mini=args.pilot)

    batch_generator = generate_batches(dataset, sampler=test_sampler, shuffle=False, 
                                       batch_size=args.test_batch_size, 
                                       device=args.device, drop_last=False)

    running_loss = 0.
    classifier.eval()
    mode = "wb"
    save_file_replace = f".csv.gz"
    save_filename = os.path.basename(args.model_state_file).replace(".pth", save_file_replace)
    save_file = os.path.join(args.save_dir, save_filename)
    
    # Runnning evaluation routine
    test_bar = tqdm.tqdm(desc=f'split={dataset_split}',
                          total=len(dataset)//args.test_batch_size, 
                          position=0, 
                          leave=True)

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())
        save_test_pred(save_file, 
                       torch.sigmoid(torch.flatten(y_pred)), 
                       batch_dict['y_target'], 
                       batch_dict["genome_loc"], 
                       mode=mode)
        mode = "ab" 

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].view(-1, 1).float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # update test bar
        test_bar.set_postfix(loss=running_loss, 
                              batch=batch_index)
        test_bar.update()
    
    return save_file

####################
# dataset creation #
####################

def read_bed_to_df(bed_file):
    df = pd.read_csv(bed_file, sep="\t", header=None, usecols=[0,1,2])
    df.columns = ["chrm", "start", "end"]
    return df

def create_tf_dataset_file(peak_bed_path, non_peak_bed_path, split_ratio=(80,10,10)):
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
