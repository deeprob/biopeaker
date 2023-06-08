import sys
import os
import subprocess
import numpy as np
import itertools
from pyfaidx import Fasta
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class GenomeVocabulary(object):
    """
    A class to store the original genome as a pyfaidx Fasta object to recover nucleotide sequences from genomic coordinates
    """
    
    def __init__(self, genome_fasta):
        """
        genome_fasta: A pyfaidx.fasta object as the genome
        """
        self.genome = genome_fasta
        
    def get_sequence(self, chrom_name, start, end):
        """A method that returns the dna sequence given their chromosomal coordinates"""
        seq = self.genome.get_seq(chrom_name, start, end)
        try:
            assert len(seq) == end-start
        # weird pyfaidx sequence length error
        except AssertionError:
            req_end = end-start
            seq = seq[0:req_end]
        return seq
    
    @classmethod
    def load_from_path(cls, genome_filepath):
        genome_fasta = Fasta(genome_filepath, as_raw=True)
        return cls(genome_fasta)

class HomerVectorizer:
    """
    Vectorizes chromosomal locations by scanning them using Homer and its motif database
    """
    def __init__(self, roi_bed, genome_fasta, homer_pwm_motifs, homer_outdir, threads=32):
        self.genome = genome_fasta
        self.homer_pwms = homer_pwm_motifs
        self.roi = roi_bed
        self.homer_outdir = homer_outdir
        self.threads = threads

        self.homer_outfile = self._get_homer_outfile()
        self.homer_roi = self._get_roi_homer()
        self.logfile = self._get_logfile()
        pass

    def _get_homer_outfile(self):
        return os.path.join(self.homer_outdir, "motif_odds.tsv")

    def _get_roi_homer(self):
        return os.path.join(self.homer_outdir, "tmp_roi_homer.bed")

    def _get_logfile(self):
        return os.path.join(self.homer_outdir, "homer_scan.log")

    def _process_homer(self, df_row):
        chrname = df_row.chrm
        start = df_row.start
        end = df_row.end
        peak_name = f"{chrname}_{start}_{end}"
        irrelevant_col = 0
        strand = "."
        return pd.Series({"3":peak_name, "4":irrelevant_col, "5": strand}) 

    def _create_homer_compatible_roi(self):
        """
        Converts an roi file with chromosomal coordinates to a homer compatible one
        """
        df_roi = pd.read_csv(self.roi, usecols=[0,1,2], sep="\t", header=None).drop_duplicates()
        df_roi.rename(columns={0: "chrm", 1: "start", 2: "end"}, inplace=True)
        df_roi.loc[:, ["chrm", "start", "end"]].merge(df_roi.apply(self._process_homer, axis=1), left_index=True, right_index=True).to_csv(self.homer_roi, index=False, header=None, sep="\t")
        return

    def _pwm_scan_homer(self):

        cmd = [
            "findMotifsGenome.pl", self.homer_roi, self.genome, self.homer_outdir, 
            "-find", self.homer_pwms, "-p", str(self.threads), "-size", "given", 
            ]
        with open(self.logfile, "w") as lf:
            with open(self.homer_outfile, "w") as of:
                results = subprocess.run(cmd, stdout=of, stderr=lf)
        return results

    def _parse_homer_outfile(self):
        motif_df = pd.read_csv(self.homer_outfile, sep="\t")
        # pivot the table for easier functioning, 
        # also for all motifs present more than once in a region scores will be summed 
        motif_df = motif_df.pivot_table('MotifScore', ["PositionID"], ["Motif Name", "Strand"], aggfunc=np.sum, fill_value=0.0)
        # fix column names and their levels after pivot table
        motif_df.columns = [f'{i}|{j}' if j != '' else f'{i}' for i,j in motif_df.columns]
        return motif_df

    def featurize(self):
        # making sure there is a directory on path
        os.makedirs(self.homer_outdir, exist_ok=True)
        # creating homer compatible tmp roi file
        self._create_homer_compatible_roi()
        # running homer
        self._pwm_scan_homer()
        # parsing the homer outfile
        self.motif_df = self._parse_homer_outfile()
        return

    def store_features_to_file(self, file_path):
        self.motif_df.to_csv(file_path)
        return

    @classmethod
    def load_features_from_file(self, file_path):
        return pd.read_csv(file_path, index_col=0)


class GenomeVectorizer(GenomeVocabulary):
    """A class that converts the chromosomal coordinates to numerical encodings"""
    def __init__(self, genome_fasta, vectorizer, **kwargs):
        """
        genome_fasta: A pyfaidx.fasta object as the genome
        vectorizer can be one of ohe, kmer or homer 
        """
        super(GenomeVectorizer, self).__init__(genome_fasta)
        # dictionary for ohe vectorizer
        self._ohe_dict = {"A":[1, 0, 0, 0],
                         "C":[0, 1, 0, 0],
                         "G":[0, 0, 1, 0],
                         "T":[0, 0, 0, 1],
                         "a":[1, 0, 0, 0],
                         "c":[0, 1, 0, 0],
                         "g":[0, 0, 1, 0],
                         "t":[0, 0, 0, 1]}
        
        self.vectorizer_dict = {"ohe": self._ohe_vectorize, "kmer": self._kmer_vectorize, "homer": self._homer_vectorize}
        
        self.vectorizer_name = vectorizer

        if vectorizer == "kmer":
            # create kmer vectorizer attributes
            self.nt = "ATGC"
            self.k = kwargs["k"]
            self._kmer2idx_dict = {"".join(k):i for i, k in enumerate(itertools.product(self.nt, repeat=self.k))}
            self._idx2kmer_dict = {v:k for k,v in self._kmer2idx_dict.items()}
            self.feature_size = (1, len(self._kmer2idx_dict))

        elif vectorizer == "homer":
            # create homer vectorizer attributes
            self.featurized_df = self.homer_featurize(**kwargs)
            self.feature_size = (1, self.featurized_df.shape[1])
        
        self.vectorize = self.vectorizer_dict[vectorizer]
                
    def _ohe_vectorize(self, chrm, start, end):
        """Vectorizer that produces one hot encoded sequence"""
        seq = self.get_sequence(chrm, start, end)
        ohe = np.array([self._ohe_dict.get(nt, [0, 0, 0, 0]) for nt in seq], dtype=np.float32)
        return np.transpose(ohe) # change from 500*4 to 4*500
    
    def _kmer_vectorize(self, chrm, start, end):
        """Vectorizer that produces normalized term frequencies of the kmer present in sequence"""
        seq = self.get_sequence(chrm, start, end)
        
        nt_vocab_len = len(self._kmer2idx_dict)
        arr = np.zeros(nt_vocab_len + 1) # 1 added to accomodate unknowns; last index of array is for unknowns
        
        for i in range(0, len(seq)- self.k + 1):
            # if kmer in dict, count increases, else unknown count increases
            arr[self._kmer2idx_dict.get(seq[i:i+self.k], nt_vocab_len)] += 1
        
        narr = arr/sum(arr)
        return narr[:-1].reshape(1, -1)

    def _homer_vectorize(self, chrm, start, end):
        """Vectorizer provided by the user"""
        seq_id = "_".join([chrm, str(start), str(end)])
        return self.featurized_df.loc[seq_id].values.reshape(1, -1)

    def homer_featurize(self, **kwargs):

        if "homer_saved" in kwargs:
            homer_saved = kwargs["homer_saved"]
            if os.path.exists(homer_saved):
                featurized_df = HomerVectorizer.load_features_from_file(homer_saved)
                print("Loaded saved file ... ")
            else:
                raise IOError("FileNotFound: Saved file does not exist!")
        else:
            # assert that the required arguments for homer featurizer is present
            roi_bed = kwargs["roi_bed"]
            homer_pwm_motifs = kwargs["homer_pwm_motifs"]
            homer_outdir = kwargs["homer_outdir"]
            # run homer to vectorize regions
            hv = HomerVectorizer(roi_bed, self.genome, homer_pwm_motifs, homer_outdir)
            hv.featurize()
            homer_saved = os.path.join(homer_outdir, "motif_features.csv.gz")
            hv.store_features_to_file(homer_saved)
            print(f"Homer gzipped csv file stored in {homer_saved}, Can be loaded directly from path next time!!!")
            featurized_df = HomerVectorizer.load_features_from_file(homer_saved)
        return featurized_df

    @classmethod
    def load_from_path(cls, genome_filepath, vectorizer, **kwargs):
        genome_fasta = Fasta(genome_filepath, as_raw=True)
        return cls(genome_fasta, vectorizer, **kwargs)
    
    
class TFDataset(Dataset):
    """A Class that contains all genomic coordinates with their labels"""
    
    def __init__(self, tf_df, vectorizer, addn_feat_df):
        """
        tf_df: A dataframe with 5 columns, chrm, start, end, label, split
        vectorizer: The object that vectorizes this dataset going from genomic coordinates to sequence to numerical encodings
        """
        
        self.tf_df = tf_df
        self._vectorizer = vectorizer
        self.addn_df = addn_feat_df
        
        self.set_split("train")
        pass
    
    @classmethod
    def load_dataset_and_vectorizer_from_path(
        cls, tf_df_path, genome_path, addn_feat_path="", nrows=None, vectorizer="ohe", k=5, homer_saved="", homer_pwm_motifs="", homer_outdir=""):
        """
        tf_df_path: path to the tf hdf5 file with genomic locations and annotations  
        genome_path: path to the genome fasta file of the organism
        addn_feat_path: path to the additional features hdf5 file
        """
        tf_df = pd.read_hdf(tf_df_path, stop=nrows).reset_index(drop=True)
        addn_df = pd.DataFrame()
        if addn_feat_path:
            addn_df = pd.read_hdf(addn_feat_path, stop=nrows).reset_index(drop=True)
            addn_df = addn_df.set_index(addn_df.columns[0])
            assert len(tf_df) == len(addn_df)
        vectorizer = GenomeVectorizer.load_from_path(genome_path, vectorizer=vectorizer, k=k, homer_saved=homer_saved, homer_pwm_motifs=homer_pwm_motifs, homer_outdir=homer_outdir, roi_bed=tf_df_path)
        return cls(tf_df, vectorizer, addn_df)
    
    def set_split(self, split="train"):
        self._target_split = split
        self._target_df = self.tf_df if self._target_split=="all" else self.tf_df[self.tf_df.split==split] 
        self._target_size = len(self._target_df)
        if not self.addn_df.empty:
            self._target_addn_df = self.addn_df.iloc[self._target_df.index]
        else:
            self._target_addn_df = pd.DataFrame()
        return
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        chrm, start, end = row.chrm, row.start, row.end
        tf_vector = self._vectorizer.vectorize(row.chrm, row.start, row.end)
        tf_label = row.label
        if not self._target_addn_df.empty:
            seq_id = "_".join([chrm, str(start), str(end)])
            addn_feat = self._target_addn_df.iloc[index]
            try:
                assert addn_feat.name==seq_id
            except AssertionError:
                raise ValueError(f"sequence id: {seq_id} and feature id: {addn_feat.name} do not match")
            addn_feat_vector = addn_feat.values.reshape(1, -1)
        else:
            addn_feat_vector = np.array([])
        return {"x_data": tf_vector,
                "a_data": addn_feat_vector,
                "y_target": tf_label,
                "genome_loc": (chrm, start, end)}
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    
    def get_feature_size(self):
        """
        Func to output the length of the feature vector created by the vectorizer
        NB: This function assumes that all genomic coordinates length in the df 
        is same as the length of the first genomic coordinate length. 
        TODO: Modify to include a check across the dataframe.
        TODO: Rewrite function to include multiple features
        """
        # length of the dna sequence is end - start
        seq_length = self._target_df.iloc[0, 2] - self._target_df.iloc[0, 1]
        return (4, seq_length) if self._vectorizer.vectorizer_name=="ohe" else self._vectorizer.feature_size


def load_data(tf_df_path, genome_fasta, vectorizer, addn_feat_path="", k=5, homer_saved="", homer_pwm_motifs="", homer_outdir=""):
    tf_dataset = TFDataset.load_dataset_and_vectorizer_from_path(tf_df_path, 
                                                              genome_fasta,
                                                              addn_feat_path=addn_feat_path,
                                                              vectorizer=vectorizer, 
                                                              k=k, homer_saved=homer_saved, homer_pwm_motifs=homer_pwm_motifs, homer_outdir=homer_outdir)
    return tf_dataset
