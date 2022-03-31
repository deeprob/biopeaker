import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

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
        peak_name = f"{chrname}:{start}-{end}"
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
            "-find", self.homer_pwms, "-p", str(self.threads)
            ]
        with open(self.logfile, "w") as lf:
            with open(self.homer_outfile, "w") as of:
                results = subprocess.run(cmd, stdout=of, stderr=lf)
        return results

    def _parse_index(self, row):
        loc_info = row.split(":")
        chrm = loc_info[0]
        start,end = loc_info[1].split("-")
        return pd.Series({"chrm": str(chrm), "start": int(start), "end": int(end)})

    def _parse_homer_outfile(self):
        motif_df = pd.read_csv(self.homer_outfile, sep="\t")
        # pivot the table for easier functioning, 
        # also for all motifs present more than once in a region scores will be summed 
        motif_df = motif_df.pivot_table('MotifScore', ["PositionID"], ["Motif Name", "Strand"], aggfunc=np.sum, fill_value=0.0)
        # fix column names and their levels after pivot table
        motif_df.columns = [f'{i}|{j}' if j != '' else f'{i}' for i,j in motif_df.columns]
        # parse the indices and return chrm, start, end
        motif_df = pd.concat(
            (motif_df.reset_index(drop=True), 
            pd.Series(motif_df.index).apply(self._parse_index)), 
            axis=1).set_index(["chrm", "start", "end"])
        return motif_df

    def featurize(self):
        # creating homer compatible tmp roi file
        self._create_homer_compatible_roi()
        # making sure there is a directory on path
        os.makedirs(self.homer_outdir, exist_ok=True)
        # running homer
        self._pwm_scan_homer()
        # parsing the homer outfile
        self.motif_df = self._parse_homer_outfile()
        return

    def store_features_to_pickle(self, pickle_path):
        self.motif_df.to_pickle(pickle_path)
        return

    @classmethod
    def load_features_from_pickle(self, pickle_path):
        return pd.read_pickle(pickle_path)


class Biopeaker:

    def __init__(self, in_beds, ko_beds):

        self.rep_num = self._get_repnum(in_beds, ko_beds)
        self.all_beds = in_beds + ko_beds
        self.df = self._convert_beds_to_df()
        self.featurized_df = pd.DataFrame()
        pass

    ################
    # file parsing #
    ################

    def _get_repnum(self, in_beds, ko_beds):
        assert len(in_beds) == len(ko_beds)
        return len(in_beds)

    def _convert_beds_to_df(self):
        df = pd.concat(list(map(self.read_and_extract_coverage, self.all_beds)), axis=1)
        df.columns = [f"{line} Rep:{i}" for line in ["Input", "Output"] for i in range(1, self.rep_num + 1)]
        return df
    
    @staticmethod
    def read_and_extract_coverage(cov_bed):
        """
        Reads the location and read depth of a region from a bed file
        Normalizes the read depth to RPKM value
        """
        df = pd.read_csv(cov_bed, sep="\t", header=None, usecols=[0,1,2,3], names=["chrm", "start", "end", "reads"])
        # assign pseudo count of 1 to 0 read regions
        df.reads = df.reads.replace(0, 1)
        # get the length of the regions to calculate rpkm values
        df_gene_length = df.end - df.start
        # calculate rpkm
        df_norm_reads = (df.reads*10**6*10**3)/(df_gene_length*df.reads.sum())
        df["rpkm"] = df_norm_reads
        df = df.set_index(["chrm", "start", "end"])
        return df.loc[:, ["rpkm"]]
    
    ############
    # plotting #
    ############

    def _get_cindex(self, num, color_list):
        return num%len(color_list)

    def _get_colors(self, val, num, pcolors, mcolors):
        if val<0:
            color = mcolors[self._get_cindex(num, mcolors)]
        else:
            color = pcolors[self._get_cindex(num, pcolors)]
        return color

    def plot_rep_fc(self):
        """
        Creates a manhattan plot of the normalized output to input log2 fold change for each replicate
        """
        fig,ax = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True) # Set the figure size
        icolors = ['darkred','darkgreen','darkblue', 'darkorange']
        ocolors = ['lightcoral', 'palegreen', 'lightskyblue', 'sandybrown']

        for i in range(len(self.df.columns)//2):

            rep_df = self.df.iloc[:, [i,i+3]]
            rep_diff = np.log2(rep_df.iloc[:,1]/rep_df.iloc[:,0])
            rep_ind = range(len(rep_df))
            rep_df["difference"] = rep_diff
            rep_df["ind"] = rep_ind
            rep_df_grouped = rep_df.groupby(level=0)
            
            x_labels = []
            x_labels_pos = []
            for num, (name, group) in enumerate(rep_df_grouped):
                colors = group.difference.apply(self._get_colors, args=(num, icolors, ocolors))
                group.plot(kind='scatter', x='ind', y='difference',color=colors, ax=ax[i], legend=False, )
                x_labels.append(name)
                x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))


            ax[i].set_xticks(x_labels_pos)
            ax[i].set_xticklabels(x_labels, rotation=45)

            # set axis limits
            ax[i].set_xlim([0, len(rep_df)])

            # x axis label
            ax[i].set_xlabel('Chromosome')
            ax[i].set_ylabel('Log2 RPKM Fold Change')
        return

    ##############
    # featurizer #
    ##############

    def homer_featurize(self, **kwargs):
        
        if "homer_pickle" in kwargs:
            homer_pickle = kwargs["homer_pickle"]
            if os.path.exists(homer_pickle):
                self.featurized_df = HomerVectorizer.load_features_from_pickle(homer_pickle)
                print("Loaded pickle ... ")
            else:
                raise IOError("FileNotFound: Pickle file does not exist!")
        else:
            # assert that the required arguments for homer featurizer is present
            roi_bed = kwargs["roi_bed"]
            genome_fasta = kwargs["genome_fasta"]
            homer_pwm_motifs = kwargs["homer_pwm_motifs"]
            homer_outdir = kwargs["homer_outdir"]
            # run homer to vectorize regions
            hv = HomerVectorizer(roi_bed, genome_fasta, homer_pwm_motifs, homer_outdir)
            hv.featurize()
            homer_pickle = os.path.join(homer_outdir, "motif_features.pkl")
            hv.store_features_to_pickle(homer_pickle)
            print(f"Homer pickle stored in {homer_pickle}, Can be loaded directly from path next time!!!")
            self.featurized_df = HomerVectorizer.load_features_from_pickle(homer_pickle)

        return

    #############
    # labelizer #
    #############

    def _get_log2_rpkm_fc(self, in_col, out_col):
        df_rep = self.df.iloc[:, [in_col, out_col]]
        df_lfc = np.log2(df_rep.iloc[:,1]/df_rep.iloc[:,0])
        return df_lfc

    def get_lfc_df(self):
        df_lfc = pd.concat(
            [self._get_log2_rpkm_fc(i, i+self.rep_num) for i in range(self.rep_num)], 
            axis=1
            )
        df_lfc.columns = [f"Rep {i}" for i in range(1, self.rep_num + 1)]
        return df_lfc

    def _annot_labels(self, val, high_thresh, low_thresh=None):
        """
        Divide regions into 4 bins
        1. High Activity: Log2 RPKM FC > 1
        2. Basal Activity: 0 < Log2 RPKM FC < 1
        3. Low Activity: -1 < Log2 RPKM FC < 0
        4. Extreme Low Activity: Log2 RPKM FC < -1
        """
        if not low_thresh:
            low_thresh = - high_thresh
        label = None
        if val>high_thresh:
            label = "high"
        elif val>0:
            label = "basal"
        elif val>low_thresh:
            label = "inactive"
        else:
            label = "low"
        return label

    def _create_numeric_labels(self, label_arr, param_dict):
        def num_lab(val):
            return param_dict[val]
        return np.array(list(map(num_lab, label_arr))) 

    def _create_categories(self, in_col, out_col, label_to_num_dict, high_thresh, low_thresh):
        # convert the df columns to their rpkm values and get log2 fold change
        df_lfc_rep = self._get_log2_rpkm_fc(in_col, out_col)
        # bin the dataframe into categories based on their fc values
        df_binned = df_lfc_rep.apply(self._annot_labels, args=(high_thresh, low_thresh))
        df_binned = df_binned[~df_binned.index.duplicated(keep='first')]
        # get rid of categories not in the label dict ones
        df_binned =  df_binned.loc[(df_binned.isin(list(label_to_num_dict.keys())))]
        return df_binned

    def _create_labels(self, df_binned, label_to_num_dict):
        y = self._create_numeric_labels(df_binned.values, label_to_num_dict)
        df_labels = df_binned.to_frame()
        df_labels["labels"] = y
        return df_labels["labels"]

    ############################
    # preprocessing model data #
    ############################

    def _create_processed_df(self, in_col, out_col, label_to_num_dict, high_thresh, low_thresh):
        """
        A function that creates a final processed dataframe which
        has chromosomal locations as index, features as the first N-1 columns 
        and labels as the final columns 
        """
        df_bin = self._create_categories(in_col, out_col, self.label_to_num_dict, high_thresh, low_thresh)
        df_label = self._create_labels(df_bin, label_to_num_dict)
        motif_df_processed = self.featurized_df.merge(df_label.to_frame(), left_index=True,  right_index=True)
        return motif_df_processed

    def _get_train_valid_split(self, motif_df_processed):
        X = motif_df_processed.iloc[:, :-1].values
        y = motif_df_processed.iloc[:, -1].values
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)
        return X_train, X_valid, y_train, y_valid

    def _get_test_data(self, motif_df_processed):
        X = motif_df_processed.iloc[:, :-1].values
        y = motif_df_processed.iloc[:, -1].values
        return X, y

    def _get_rep_processed_dfs(self, high_thresh, low_thresh):
        """
        Creates processed dataframes for all replicates
        """
        processed_rep_dfs = [
            self._create_processed_df(
                i, i+self.rep_num, self.label_to_num_dict, high_thresh, low_thresh, 
                ) for i in range(self.rep_num)
            ]
        return processed_rep_dfs

    def _get_train_valid_test(self, high_thresh, low_thresh):
        pr_dfs = self._get_rep_processed_dfs(high_thresh, low_thresh)
        X_train, X_valid, y_train, y_valid = self._get_train_valid_split(pr_dfs[0])

        test_data = []
        if len(pr_dfs)>1:
            test_data = [self._get_test_data(pr_df) for pr_df in pr_dfs[1:]]    
        return X_train, X_valid, y_train, y_valid, test_data

    ##########
    # models #
    ##########

    def _get_linear_model(self, X_train, y_train):
        cw = {c:1/freq for c,freq in  dict(zip(*np.unique(y_train, return_counts=True))).items()}
        linear_baseline = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight=cw, penalty="l2", solver="lbfgs", C=1))
        linear_baseline.fit(X_train, y_train)
        return linear_baseline

    ############
    # evaluate #
    ############

    def _get_sklearn_model_preds(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def _get_acc_scores(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        pr = precision_score(y_test, y_pred)
        re = recall_score(y_test, y_pred)
        return acc, bacc, pr, re

    def _get_activators_and_repressors(self, linear_model):
        map_tuple = tuple(zip(list(self.featurized_df.columns),linear_model['logisticregression'].coef_[0]))
        activators = [(tf, val) for tf,val in sorted(map_tuple, key=lambda x: abs(x[1]), reverse=True) if val>0]
        repressors = [(tf, val) for tf,val in sorted(map_tuple, key=lambda x: abs(x[1]), reverse=True) if val<0]
        return activators, repressors

    ##########
    # peaker #
    ##########

    def peaker(
        self, 
        labels_of_interest=["low", "high"],
        high_thresh=1.5, low_thresh=-1.5, 
        ):
        # set the label to num dict used for creating numerical labels for enhancer categories
        self.label_to_num_dict = dict(zip(labels_of_interest, range(len(labels_of_interest))))

        # check if featurized
        if self.featurized_df.empty:
            raise ValueError("Regions are not featurized: Please featurize before calling peaker")
        
        # get train-valid-test data
        X_train, X_valid, y_train, y_valid, test_data = self._get_train_valid_test(high_thresh, low_thresh)

        # train linear model
        linear_model = self._get_linear_model(X_train, y_train)

        # evaluate on validation
        valid_pred = self._get_sklearn_model_preds(linear_model, X_valid)
        valid_evaluation = self._get_acc_scores(valid_pred, y_valid)
        print(valid_evaluation)
        a, r = self._get_activators_and_repressors(linear_model)
        return a, r, valid_pred
        
