from ModelTrainer.modelStructures import GATWithBERT
from ModelTrainer.trainer import train
from ModelTrainer.graphFunctions import set_seed
import random
from Tester.generalizer import generalization_test
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("GPU count:", torch.cuda.device_count())
    seeds = []
    RUNS_PER_TRAINING = 2 # TODO: It's 5 instead
    BATCH_SIZE = 32
    WORKERS = 8

    for run in range(1, RUNS_PER_TRAINING+1): # Five runs per training, to show it isn't a lucky seed 
        seed = random.randint(1, 101)
        while seed in seeds: # Avoid repeated seeds
            seed = random.randint(1, 101)
        set_seed(seed) # Reset the seed every run
        seeds.append(seed)
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            # Train every dataset with the same seed before moving to another
            model = GATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
            train(model, dataset, epochs=3, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/GAT {run} (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/GAT {run} (seed {seed}).txt", run=run)
            generalization_test(model, dataset, calling_run=run, seed=seed, batch_size=BATCH_SIZE)

    # TODO: Pensar em quais serão as ablações
    # Ablation
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = GATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=12) # Transfer learning ablation
            train(model, dataset, epochs=1, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).txt", run=run, ablation=True)
            generalization_test(model, dataset)