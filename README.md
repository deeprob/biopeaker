# Deep learning methods to classify genomic regions
Train, predict, evaluate and interpret deep learning models to classify any genomic region that peaked (had significantly higher mapped reads compared to other regions) in any sequencing based functional dataset such as ChIP-seq, ATAC-seq, STARR-seq.

# Quickstart
## Clone repo
```bash
    $ git clone https://github.com/deeprob/biopeaker.git
```

## Create conda env
```bash
    $ conda_env_path="/path/to/dlenv"
    $ conda create --prefix $conda_env_path -c conda-forge -c anaconda -c bioconda python=3.7 homer
    $ conda activate $conda_env_path
    $ pip3 install -r requirements.txt
```

## Create peaker compatible dataset
```bash
    $ peak_bed_file="/path/to/peaks.bed"
    $ nonpeak_bed_file="/path/to/notpeaks.bed"
    $ dataset_path="/path/to/dataset.csv"
    $ create_dataset_path="/path/to/create_dataset.py"
    $ python $create_dataset_path $peak_bed_file $nonpeak_bed_file $dataset_path
```

## Run model with default settings
Will use Resnet encoder and MLP classifier
```bash
    $ peaker_path="/path/to/peaker.py"
    $ dataset_path="/path/to/dataset.csv"
    $ genome_fasta="/path/to/genome_fasta.fa"
    $ save_dir="/path/to/save_dir/"
    $ python $peaker_path $dataset_path $genome_fasta $save_dir
```

## Eval model
```bash
    $ peaker_path="/path/to/peaker.py"
    $ dataset_path="/path/to/dataset.csv"
    $ genome_fasta="/path/to/genome_fasta.fa"
    $ save_dir="/path/to/save_dir/"
    $ python $peaker_path $dataset_path $genome_fasta $save_dir --test
```

# Compatible encoders
1. residual networks a.k.a resnet
2. homer motif scan a.k.a homer

# Compatible classifiers
1. Logistic Regression a.k.a linear
2. Multi Layer Perceptron a.k.a mlp


# Interpretation tools
1. Integrated Gradients

# Major changes from v1.0.0
1. Model separated into encoder and classifier
2. Peaker interface changed, by default resnet with mlp is chosen method
3. Peaker dataset type changed from csv to hdf5 format
4. Integrated gradients calculation now included for linear model
5. Encoder freezing capability added
6. Additional features beyond sequence can now to added
