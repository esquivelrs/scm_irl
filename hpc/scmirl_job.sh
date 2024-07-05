#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J scmirl_gailrep
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 48:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=15GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s230025@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.8
module load gcc

#python3 -m pip install tensorflow[and-cuda]
python3
#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/11.8.0
#python3 -m pip install tensorflow-compression
#python3 -m pip install easydict
#python3 -m pip install tensorflow==2.14.0

source /zhome/11/1/193832/miniconda3/bin/activate

conda activate scm

python /zhome/11/1/193832/resquivel/scm_irl/scripts/train_irl.py
