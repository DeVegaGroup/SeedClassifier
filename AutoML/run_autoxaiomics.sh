#!/bin/bash
#SBATCH --mem 100G -c32 -p ei-medium
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=eloise.barret@earlham.ac.uk #send-to address
#SBATCH -t 04-00:00 # time (D-HH:MM) 

exec /ei/.project-scratch/e/e88c153a-c2ee-48ea-b637-e8aca3f4d3ca/autoxai4omics_dir/autoxai4omics-1.0.0.img autoxai4omics.sh -m predict -c seed_classifier.json
 