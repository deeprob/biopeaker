import os
import argparse
import utils.helpers as uth


def call_peaker(args):
    # Set seed for reproducibility
    uth.set_seed_everywhere(args.seed, args.cuda)
    # handle dirs
    uth.handle_dirs(args.save_dir)
    # training or evaluation
    if args.train:
        train_state = uth.train_model(args)
    else:
        pred_save_file = uth.eval_model(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning based peak prediction pipelin')
    # required arguments
    parser.add_argument("model_name", type=str, help="The type of model to assign :: one of resnet, mlp, linear")
    parser.add_argument("dataset", type=str, help="The path to the parsed dataset file -  must be according to convention")
    parser.add_argument("vectorizer", type=str, help="The type of vectorizer to use :: one of ohe,kmer,homer depending on the model")
    parser.add_argument("genome_fasta", type=str, help="Path to genome fasta file")
    parser.add_argument("save_dir", type=str, help="Path where models and results will be saved")
    # other vectorizer arguments
    parser.add_argument("--homer_saved", type=str, help="Path to the homer saved file", default="")
    parser.add_argument("--homer_pwm_motifs", type=str, help="Path to the homer pwm motif file", default="")
    parser.add_argument("--homer_outdir", type=str, help="Path to the homer outputdir", default="")
    parser.add_argument("--kmer", type=int, help="Length of kmer to create", default=5)
    # model training arguments
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train", default=100)
    parser.add_argument("--pytorch_device", type=str, help="Type of pytorch device to use :: one of cuda or cpu", default="cuda")
    parser.add_argument("--dropout_prob", type=float, help="Dropout probability of neural network", default=0.25)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=64)
    parser.add_argument("--learning_rate", type=float, help="Optimizer learning rate", default=0.005)
    parser.add_argument("--early_stopping_criteria", type=float, help="Early stopping steps", default=10)
    parser.add_argument("--tolerance", type=float, help="tolerance for early stopping", default=1e-4)
    parser.add_argument("--random_seed", type=int, help="Number of epochs to train", default=7)
    parser.add_argument("--test_batch_size", type=int, help="Batch size for evaluation", default=500)
    parser.add_argument("--pilot", help="Whether it is a pilot study")
    parser.add_argument("--test", help="Evaluate only - model will not train", action="store_true")

    cli_args = parser.parse_args()
    model_args = uth.initialize_from_cli(cli_args)

    call_peaker(model_args)
