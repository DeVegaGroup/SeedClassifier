#!/bin/bash
#SBATCH --mem 100G -c8 -p ei-medium

singularity exec /ei/projects/6/640a388a-14a6-4ec5-b8c6-4242ca927a9a/singularities/autoxai2_CV.img autoxai4omics.sh -m predict -c seed_classifier_JA.json
