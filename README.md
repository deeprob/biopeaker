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

## Run model
```bash
    $ peaker_path="/path/to/peaker.py"
    $ dataset_path="/path/to/dataset.csv"
    $ genome_fasta="/path/to/genome_fasta.fa"
    $ save_dir="/path/to/save_dir/"
    $ python $peaker_path resnet $dataset_path ohe $genome_fasta $save_dir
```

## Eval model
```bash
    $ peaker_path="/path/to/peaker.py"
    $ dataset_path="/path/to/dataset.csv"
    $ genome_fasta="/path/to/genome_fasta.fa"
    $ save_dir="/path/to/save_dir/"
    $ python $peaker_path resnet $dataset_path ohe $genome_fasta $save_dir --test
```

# Training methods
1. Logistic Regression a.k.a linear
2. Multi Layer Perceptron a.k.a mlp
3. Residual Network a.k.a resnet

# Genome vectorizer methods
1. One hot encoding a.k.a ohe - compatible with resnet
2. k-mer creation a.k.a kmer - compatible with linear and mlp
3. homer motif scan a.k.a homer - compatible with linear and mlp

# Interpretation tools
1. Integrated Gradients
