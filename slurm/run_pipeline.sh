#!/bin/bash
#SBATCH --account=girirajan # TODO: set account name
#SBATCH --partition=girirajan # TODO: set slurm partition
#SBATCH --job-name=peaker 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=400:0:0
#SBATCH --mem-per-cpu=10G
#SBATCH --chdir /data6/deepro/computational_pipelines/dl_setup/data # TODO: set dir to data dir
#SBATCH -o /data6/deepro/computational_pipelines/biopeaker/slurm/logs/out_run.log # TODO: set slurm output file
#SBATCH -e /data6/deepro/computational_pipelines/biopeaker/slurm/logs/err_run.log # TODO: set slurm input file
#SBATCH --nodelist=laila
#SBATCH --gpus=1


echo `date` starting job on $HOSTNAME

peaker_path="/data6/deepro/computational_pipelines/biopeaker/src/peaker.py"
dataset_path="/data6/deepro/computational_pipelines/dl_setup/data/sample.csv"
genome_fasta="/data5/deepro/genomes/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
save_dir="/data6/deepro/computational_pipelines/dl_setup/data/resnet_new"

python $peaker_path resnet $dataset_path ohe $genome_fasta $save_dir

echo `date` ending job on $HOSTNAME
