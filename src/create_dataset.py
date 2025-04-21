import argparse
import utils.helpers as uth


def create_peaker_compatible_dataset(peak_bed, non_peak_bed, save_file, window_size, split_ratio):
    tf_df = uth.create_tf_dataset_file(peak_bed, non_peak_bed, split_ratio=split_ratio)
    tf_df = tf_df.loc[(tf_df.end-tf_df.start)==window_size]
    assert len(tf_df)>0
    tf_df.to_hdf(save_file, index=False, key="samples")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning based peak prediction pipeline')
    # required arguments
    parser.add_argument("peak_bed", type=str, help="Bed file path with genomic locations of the peaks")
    parser.add_argument("non_peak_bed", type=str, help="Bed file path with genomic locations of the non peaks")
    parser.add_argument("save_file", type=str, help="Filepath to store the created dataset - must end with h5 extension")
    # optional arguments
    parser.add_argument("--window_size", type=int, help="size of the genomic regions", default=1000)
    parser.add_argument("--split_ratio", nargs='+', type=int, help="three way split of the samples (train,valid,test)", default=[60,30,10])

    cli_args = parser.parse_args()
    create_peaker_compatible_dataset(cli_args.peak_bed, cli_args.non_peak_bed, cli_args.save_file, cli_args.window_size, cli_args.split_ratio)   
