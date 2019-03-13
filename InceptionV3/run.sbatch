#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=ALL                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=inceptionV3_train_20valid
#SBATCH --time=40:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=40960M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=inceptionV3_train_20valid.output     #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122821401817       #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=randyardywibowo@tamu.edu #Send all emails to email_address

#First Executable Line
cd $SCRATCH/inceptionV3/
module load Keras/2.1.3-goolfc-2017b-Python-3.6.3
module load OpenCV/3.3.0-foss-2017b-Python-3.6.3
module load scikit-learn/0.19.1-goolfc-2017b-Python-3.6.3
python inception_v3.py
