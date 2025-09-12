import os
import logging
import copy

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

def make_train_state(args, task_name):
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
            'encoder_filename': args.encoder_state_file,
            'classifier_filename':args.classifier_state_file[task_name]}

def update_train_state(args, encoder, classifier, train_state):
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
        if args.encoder:
            torch.save(encoder.state_dict(), train_state['encoder_filename'])
        torch.save(classifier.state_dict(), train_state['classifier_filename'])
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
                if args.encoder:
                    if args.train_encoder:
                        torch.save(encoder.state_dict(), train_state['encoder_filename'])
                torch.save(classifier.state_dict(), train_state['classifier_filename'])
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


def train_mlt_model(args):
    # Logger config
    logging.basicConfig(filename=os.path.join(args.save_dir, f"{args.encoder_name}_{args.classifier_name}.log"), filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',)
    
    # Loading the dataset
    logging.debug(f'Loading dataset and creating vectorizer...')
    dataset = load_data(args.dataset, args.genome_fasta, args.vectorizer, args.task_names, addn_feat_path=args.addn_feat_dataset, k=args.k, homer_saved=args.homer_saved, homer_pwm_motifs=args.homer_pwm_motifs, homer_outdir=args.homer_outdir)    
    logging.debug(f'Dataset loaded.')

    # Initializing encoder
    logging.debug(f'Initializing encoder...')
    if args.encoder:
        encoder = args.encoder()
        encoder_params = get_n_params(encoder)
        logging.debug(f"The encoder has {encoder_params} parameters.")
        if os.path.exists(args.encoder_state_file):
            logging.debug(f'Loading previous encoder found on path...')
            encoder.load_state_dict(torch.load(args.encoder_state_file))
            init_encoder_dict = copy.deepcopy(encoder.state_dict())
        encoder = encoder.to(args.device)
        logging.debug(f'Encoder initialized on {args.device}.')

    # Initializing multiple classifiers
    logging.debug(f'Initializing multiple classifiers...')
    classifiers = {}
    for task_name in args.task_names:
        classifiers[task_name] = args.classifier(args)  # same architecture, different instance
        classifier_params = get_n_params(classifiers[task_name])
        logging.debug(f"The classifier for {task_name} has {classifier_params} parameters.")
        # TODO: pick state files for each task
        if os.path.exists(args.classifier_state_file[task_name]):
            logging.debug(f'Loading previous classifier found on path...')
            classifiers[task_name].load_state_dict(torch.load(args.classifier_state_file[task_name])) 
    classifiers = nn.ModuleDict({k: v.to(args.device) for k, v in classifiers.items()})
    logging.debug(f'Models initialized on {args.device}.')


    # Defining loss function, optimizer and scheduler
    loss_func = nn.BCEWithLogitsLoss()
    if args.encoder:
        if args.train_encoder:
            enc_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-7)
            # adjusting the learning rate for better performance
            enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=enc_optimizer,
                                                            mode='min', factor=0.5,
                                                            patience=1)
    # Classifier optimizers and schedulers (one per task)
    cls_optimizers = {}
    cls_schedulers = {}
    for task_name, classifier in classifiers.items():
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, eps=1e-7)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min', factor=0.5, patience=1
        )
        cls_optimizers[task_name] = optimizer
        cls_schedulers[task_name] = scheduler

    logging.debug(f"Model learning rate started with {args.learning_rate}.")


    # Making samplers
    train_sampler, valid_sampler = make_train_samplers(dataset, args)
    logging.debug(f"Training model with {train_sampler.num_samples} instances.")
    logging.debug(f"Model batch size set to {args.batch_size}.")

    # Defining initial train state
    train_state = {tn: make_train_state(args, tn) for tn in args.task_names}
    
    ##### Training Routine #####
    
    try:
        for epoch_index in range(args.num_epochs):
            for tn in args.task_names:
                train_state[tn]['epoch_index'] = epoch_index

            # Iterate over training dataset
            logging.debug(f"Model Training --- Epoch {epoch_index}.")

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, sampler=train_sampler,
                                               batch_size=args.batch_size, 
                                               device=args.device)
            running_loss = 0.0
            if args.encoder:
                if args.train_encoder:
                    encoder.train()
                else:
                    encoder.eval()
            for classifier in classifiers.values():
                classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):

                # the training routine as follows:
                # --------------------------------------
                # step 1. zero the gradients
                if args.encoder:
                    if args.train_encoder:
                        enc_optimizer.zero_grad()
                
                for cls_optimizer in cls_optimizers.values():
                    cls_optimizer.zero_grad()

                # step 2. compute the output
                seq_feats = batch_dict['x_data'].float()
                add_feats = batch_dict['a_data'].float()
                if args.encoder:
                    seq_feats = encoder(x_in=seq_feats)
                    if not args.train_encoder:
                        seq_feats = seq_feats.detach()
                        assert seq_feats.requires_grad == False

                losses = []
                for tn in args.task_names:
                    classifier = classifiers[tn]
                    y_pred = classifier(x_in=seq_feats, a_in=add_feats)
                    loss = loss_func(y_pred, batch_dict[tn].view(-1, 1).float())
                    losses.append(loss)

                # step 3. compute the loss
                total_loss = sum(losses)

                # step 4. use loss to produce gradients
                total_loss.backward()

                # step 5. use optimizer to take gradient step
                if args.encoder and args.train_encoder:
                    enc_optimizer.step()
                for cls_optimizer in cls_optimizers.values():
                    cls_optimizer.step()
                # -----------------------------------------

                # compute the loss for update
                loss_t = total_loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

            for tn in args.task_names:
                train_state[tn]['train_loss'].append(running_loss)

            # Iterate over val dataset
            logging.debug(f"Model Evaluation --- Epoch {epoch_index}.")

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('valid')
            batch_generator = generate_batches(dataset, sampler=valid_sampler,
                                               batch_size=int(args.test_batch_size), 
                                               device=args.device)

            if args.encoder:
                encoder.eval()

            for classifier in classifiers.values():
                classifier.eval()

            tmp_filenames = {tn: os.path.join(args.save_dir, f"validation_{tn}.tmp") for tn in args.task_names}
            tmp_files = {tn: open(fname, "wb") for tn, fname in tmp_filenames.items()}
            running_losses = {tn: 0.0 for tn in args.task_names}

            for batch_index, batch_dict in enumerate(batch_generator):

                seq_feats = batch_dict['x_data'].float()
                add_feats = batch_dict['a_data'].float()

                if args.encoder:
                    seq_feats = encoder(x_in=seq_feats)

                for tn in args.task_names:
                    classifier = classifiers[tn]
                    y_pred = classifier(x_in=seq_feats, a_in=add_feats)
                    y_target = batch_dict[tn].view(-1, 1).float()

                    loss = loss_func(y_pred, y_target)
                    loss_t = loss.item()
                    running_losses[tn] += (loss_t - running_losses[tn]) / (batch_index + 1)

                    for yp, yt in zip(torch.sigmoid(torch.flatten(y_pred)).cpu().detach().numpy(), torch.flatten(y_target).cpu().detach().numpy()):
                        tmp_files[tn].write(bytes(f"{yp},{yt}\n", "utf-8"))

            for tn in args.task_names:
                tmp_files[tn].close()

            # Compute val loss and aps for each task
            for tn in args.task_names:
                val_aps = compute_aps_from_file(tmp_filenames[tn])
                os.remove(tmp_filenames[tn])

                train_state[tn]['val_loss'].append(running_losses[tn])
                train_state[tn]['val_aps'].append(val_aps)

                train_state[tn] = update_train_state(args=args,
                                                    encoder=encoder if args.encoder else None,
                                                    classifier=classifiers[tn],
                                                    train_state=train_state[tn])

            # schedulers step
            if args.encoder and args.train_encoder:
                enc_scheduler.step(sum([train_state[tn]['val_loss'][-1] for tn in args.task_names]) / len(args.task_names))

            for tn in args.task_names:
                cls_schedulers[tn].step(train_state[tn]['val_loss'][-1])

            logging.debug(f"Epoch {epoch_index} summary:")
            for tn in args.task_names:
                logging.debug(f"Task {tn}: Validation Loss: {running_losses[tn]}, Validation APS: {train_state[tn]['val_aps'][-1]}, Early stopping step: {train_state[tn]['early_stopping_step']}")

            # early stopping: stop if all tasks want to stop
            if all(train_state[tn]['stop_early'] for tn in args.task_names):
                logging.debug("All tasks fulfilled early stopping criterion!")
                break

            # verify encoder unchanged if frozen
            if not args.train_encoder:
                for t1, t2 in zip(init_encoder_dict.values(), encoder.state_dict().values()):
                    assert torch.equal(t1, t2.cpu())

    except KeyboardInterrupt:
        logging.warning("Exiting loop")

    return train_state
