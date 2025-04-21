#!/bin/bash
#SBATCH --account=girirajan # TODO: set account name
#SBATCH --partition=girirajan # TODO: set slurm partition
#SBATCH --job-name=peaker 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=400:0:0
#SBATCH --mem-per-cpu=10G
#SBATCH --chdir /data7/deepro/pipelines/biopeaker/data/tmp # TODO: set dir to data dir
#SBATCH -o /data7/deepro/pipelines/biopeaker/slurm/logs/out_run.log # TODO: set slurm output file
#SBATCH -e /data7/deepro/pipelines/biopeaker/slurm/logs/err_run.log # TODO: set slurm input file
#SBATCH --nodelist=laila
#SBATCH --gpus=1

export HOME="/data7/deepro/pipelines/biopeaker/data/tmp"
echo `date` starting job on $HOSTNAME

peaker_path="/data7/deepro/pipelines/biopeaker/src/peaker.py"
dataset_path="/data7/deepro/pipelines/biopeaker/data/tmp/test_1000.h5"
genome_fasta="/data5/deepro/genomes/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
save_dir="/data7/deepro/pipelines/biopeaker/data/tmp/resnet_1000"

python $peaker_path $dataset_path $genome_fasta $save_dir # --pilot

echo `date` ending job on $HOSTNAME
