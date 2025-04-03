#!/bin/bash
#SBATCH --mem 30G -c8 -p ei-long -J gluon

##singularity exec /ei/projects/e/e88c153a-c2ee-48ea-b637-e8aca3f4d3ca/pytorch20240709.img python /ei/.project-scratch/e/e88c153a-c2ee-48ea-b637-e8aca3f4d3ca/scripts/autogluon.py

singularity exec /ei/projects/e/e88c153a-c2ee-48ea-b637-e8aca3f4d3ca/pytorch20240709.img python ./autogluon_train_jdv-rerun.py


#singularity exec /ei/projects/e/e88c153a-c2ee-48ea-b637-e8aca3f4d3ca/pytorch20240709.img python ./autogluon_train_balanced_jdv-rerun.py


