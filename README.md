# Deep learning method to classify genomic regions

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
