#!/bin/bash
#SBATCH --account=girirajan # TODO: set account name
#SBATCH --partition=girirajan # TODO: set slurm partition
#SBATCH --job-name=peaker 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=400:0:0
#SBATCH --mem-per-cpu=10G
#SBATCH --chdir /data7/deepro/pipelines/biopeaker/data/tmp # TODO: set dir to data dir
#SBATCH -o /data7/deepro/pipelines/biopeaker/slurm/logs/out_interpret.log # TODO: set slurm output file
#SBATCH -e /data7/deepro/pipelines/biopeaker/slurm/logs/err_interpret.log # TODO: set slurm input file
#SBATCH --nodelist=laila
#SBATCH --gpus=1


export HOME="/data7/deepro/pipelines/biopeaker/data/tmp"
echo `date` starting job on $HOSTNAME
source /opt/anaconda/bin/activate /data7/deepro/miniconda3/envs/dlinterpret


ohe_path="/data7/deepro/pipelines/biopeaker/data/tmp/resnet_1000/seq_feat.npy"
attr_path="/data7/deepro/pipelines/biopeaker/data/tmp/resnet_1000/seq_attr.npy"
out_path="/data7/deepro/pipelines/biopeaker/data/tmp/resnet_1000/modisco.h5"
motif_path="/data7/deepro/starrseq/4_ml_classification_fragment_category/data/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme"
result_dir="/data7/deepro/pipelines/biopeaker/data/tmp/resnet_1000/modisco_results/"

modisco motifs -s $ohe_path -a $attr_path -n 2000 -o $out_path -w 1000
modisco report -i $out_path -o $result_dir -s $result_dir -t -n 3 -m $motif_path

echo `date` ending job on $HOSTNAME
