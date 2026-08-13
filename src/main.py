from ModelTrainer.modelStructures import GATWithBERT, GraphDataset, SyntacticGAT, SyntacticGraphDataset, SemanticOnlyBERT, TextDataset
from ModelTrainer.trainer import train
from ModelTrainer.graphFunctions import set_seed
from Tester.generalizer import generalization_test
import torch
import random

if __name__ == "__main__":
    print("GPU count:", torch.cuda.device_count())
    seeds = []
    RUNS_PER_TRAINING = 3
    BATCH_SIZE = 32
    WORKERS = 16
    print("INFO: The tokenizer will be downloaded once per worker. Expect WORKERS+1 calls for hugging face")
    print("INFO: Longer sequence lengths in tokenization are expected and treated within the code")

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

    # Ablation - Transfer Learning
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = GATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=12) # Transfer learning ablation
            train(model, dataset, epochs=3, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).txt", run=run, ablation=True)
            generalization_test(model, dataset, calling_run=run, runs=RUNS_PER_TRAINING, seed=seed, batch_size=BATCH_SIZE, ablation=True)

    # Ablation - Syntactic-only
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = SyntacticGAT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2) 
            train(model, dataset, epochs=3, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/Syntactic-only GAT {run+1} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/Syntactic-only {run+1} Ablation (seed {seed}).txt", run=run+1, ablation=True, dataset_class=SyntacticGraphDataset)
            generalization_test(model, dataset, calling_run=run+1, runs=RUNS_PER_TRAINING, seed=seed, batch_size=BATCH_SIZE, ablation=True, dataset_class=SyntacticGraphDataset)

    # Ablation - Semantic-only
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = SemanticOnlyBERT(num_classes=2, dropout_rate=0.2, freeze_bert_layers=6) # Transfer learning ablation
            train(model, dataset, epochs=3, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/Semantic-only GAT {run+1} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/Semantic-only {run+1} Ablation (seed {seed}).txt", run=run+1, ablation=True, dataset_class=TextDataset)
            generalization_test(model, dataset, calling_run=run+1, runs=RUNS_PER_TRAINING, seed=seed, batch_size=BATCH_SIZE, ablation=True, dataset_class=TextDataset)