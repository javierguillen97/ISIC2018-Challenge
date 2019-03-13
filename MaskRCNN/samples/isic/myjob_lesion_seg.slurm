#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample2       #Set the job name to "JobExample2"
#SBATCH --time=48:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8          #Request 8 tasks/cores per node
#SBATCH --mem=40G                     #Request 8GB per node 
#SBATCH --output=SimOut_lesion_seg.%j           #Send stdout/err to "Sim2Out.[jobID]" 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#User's Executable Line
module load Anaconda/3-5.0.0.1

source activate ml
python lesion_seg.py train --imagedir=/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1-2_Training_Input/ --maskdir=/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth/ --weights=coco
source deactivate

python lesion_seg.py train --imagedir=/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1-2_Training_Input/ --maskdir=/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth/ --weights=../../mask_rcnn_weights.h5
