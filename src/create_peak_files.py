import argparse
import utils.helpers as uth

CHROM_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chrX": 156040895,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr11": 135086622,
    "chr10": 133797422,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr20": 64444167,
    "chr19": 58617616,
    "chrY": 57227415,
    "chr22": 50818468,
    "chr21": 46709983
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning based peak prediction pipeline')
    # required arguments
    parser.add_argument("peak_bed", type=str, help="Bed file path with genomic locations of the peaks")
    parser.add_argument("save_dir", type=str, help="Filepath to store peak and non peak bed files")
    # optional arguments
    parser.add_argument("--window_size", type=int, help="window size of the centered regions", default=1000)
    parser.add_argument("--overlap_frac", type=float, help="maximium overlap fraction of regions to not consider as peak", default=0.5)

    cli_args = parser.parse_args()
    uth.generate_peak_files(cli_args.peak_bed, cli_args.window_size, CHROM_SIZES, cli_args.save_dir, cli_args.overlap_frac)
