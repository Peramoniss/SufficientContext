#!/bin/bash
#!Python 3.14.3
#SBATCH --job-name=sufficiency_training    # Name of the job
#SBATCH --output=result_%j.out       # Standard output log (%j = JobID)
#SBATCH --error=error_%j.err         # Error log
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=1            # CPU cores per task
#SBATCH --mem=32G                     # Memory limit (e.g., 1 Gigabyte)
#SBATCH --time=23:59:59              # Time limit (hrs:min:sec)
#SBATCH --partition=tupi         # Partition/queue name

# Load required modules (varies by cluster)
#module load python/3.14.3               

# Execute the script - DE DENTRO DA PASTA src
python3 main.py