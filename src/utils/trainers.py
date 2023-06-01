import os
import logging

import pandas as pd

from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim

from .samplers import make_train_samplers, generate_batches
from .datasets import load_data


##################
# training utils #
##################

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
    loss_dict = {
        "val_aps": {"worse": lambda x,y: x<=y , "better": lambda x,y: x>y},
        "val_loss": {"worse": lambda x,y: x>=y , "better": lambda x,y: x<y},
        }
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        curr_aps = train_state[args.early_stopping_function][0]
        train_state['early_stopping_best_val'] = curr_aps
        train_state['stop_early'] = False
    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        apc_tm1, apc_t = train_state[args.early_stopping_function][-2:] # looking at the last two validation aps
        # If loss worsened 
        if loss_dict[args.early_stopping_function]["worse"](apc_t, train_state['early_stopping_best_val']):
            # Update step
            train_state['early_stopping_step'] += 1 # updating early stopping info
        # apc increased
        else:
            # Save the best model
            if loss_dict[args.early_stopping_function]["better"](apc_t, train_state['early_stopping_best_val']):
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = apc_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0
        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria
    return train_state

def compute_aps_from_file(file):
    """Computes the aps score from a file"""
    results_df = pd.read_csv(file, header=None)
    y_target = results_df[1].values
    y_pred = results_df[0].values
    return average_precision_score(y_target, y_pred)

def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


############
# training #
############

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
        logging.debug(f'Loading previous model found on path...')
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
            
            logging.debug(f"Epoch: {epoch_index}, Validation Loss: {running_loss}, Validation APS: {val_aps}, Early stopping step: {train_state['early_stopping_step']}")

            if train_state['stop_early']:
                logging.debug("Early stopping criterion fulfilled!")
                break

    except KeyboardInterrupt:
        logging.warning("Exiting loop")
    
    return train_state
