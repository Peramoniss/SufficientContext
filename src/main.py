from ModelTrainer.modelStructures import GATWithBERT
from ModelTrainer.trainer import train
from ModelTrainer.graphFunctions import set_seed
import random
from Tester.generalizer import generalization_test

seeds = []
RUNS_PER_TRAINING = 2 # TODO: It's 5 instead
for run in range(1, RUNS_PER_TRAINING+1): # Five runs per training, to show it isn't a lucky seed 
    seed = random.randint(1, 101)
    set_seed(seed) # Reset the seed every run
    seeds.append(seed)
    for dataset, vs in [("2WikiMultihopQA", 5000), ("HotpotQA", 2500), ("MuSiQue", 2500)]: # Different validation steps for each dataset since each has different length
        # Train every dataset with the same seed before moving to another
        model = GATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
        train(model, dataset, epochs=1, batch_size=16, validation_steps=vs, model_save_path=f"../Models/{dataset}/GAT {run} (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/GAT {run} (seed {seed}).txt", run=run)
        generalization_test(model, dataset, calling_run=run, seed=seed)

# TODO: Pensar em quais serão as ablações
# Ablation
for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
    set_seed(seed) # Reset the seed every run
    for dataset, vs in [("2WikiMultihopQA", 5000), ("HotpotQA", 2500), ("MuSiQue", 2500)]: # Different validation steps for each dataset since each has different length
        model = GATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=12) # Transfer learning ablation
        train(model, dataset, epochs=1, batch_size=16, validation_steps=vs, model_save_path=f"../Models/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).txt", run=run, ablation=True)
        generalization_test(model, dataset)