from ModelTrainer.modelStructures import SyntacticGATWithBERT, GraphDataset, SyntacticGAT, SyntacticGraphDataset, SemanticOnlyBERT, TextDataset, DocGATWithBERT, DocGraphDataset, NodeDocData
from ModelTrainer.trainer import train
from ModelTrainer.graphFunctions import set_seed
from Tester.generalizer import generalization_test
from ShortcutIdentifier.shortcutBenchmarks import shortcut_validate
import torch
from tqdm import tqdm
import random

if __name__ == "__main__":

    print("GPU count:", torch.cuda.device_count())
    seeds = [4, 24, 42, 37, 2]
    ultra_prohibited_seeds = [] # Used to be able to run in different machines
    RUNS_PER_TRAINING = 1
    GENERALIZATION_RUNS = 3
    BATCH_SIZE = 16
    EPOCHS = 3
    WORKERS = 4
    print("INFO: The tokenizer will be downloaded once per worker. Expect WORKERS+1 calls for hugging face")
    print("INFO: Longer sequence lengths in tokenization are expected and treated within the code")

    for _ in range(1, RUNS_PER_TRAINING+1-len(seeds)):
        seed = random.randint(1, 101)
        while seed in ultra_prohibited_seeds: # Avoid repeated seeds
            seed = random.randint(1, 101)
        seeds.append(seed)
        ultra_prohibited_seeds.append(seed) # Prohibit from repeating that seed

    for run, seed in enumerate(seeds): # Five runs per training, to show it isn't a lucky seed 
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 5000), ("HotpotQA", 2500), ("MuSiQue", 2500)]: # Different validation steps for each dataset since each has different length
            # Train every dataset with the same seed before moving to another
            model = SyntacticGATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
            train(model, dataset, seed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/GAT {run} (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/GAT {run} (seed {seed}).txt", run=run)
            generalization_test(model, dataset, calling_run=run, seed=seed, batch_size=BATCH_SIZE, runs=GENERALIZATION_RUNS)

    # Ablation - Transfer Learning
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = SyntacticGATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=12) # Transfer learning ablation
            train(model, dataset, seed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/GAT {run} Ablation (seed {seed}).txt", run=run, ablation=True)
            generalization_test(model, dataset, calling_run=run, runs=GENERALIZATION_RUNS, seed=seed, batch_size=BATCH_SIZE, ablation=True)

    # Ablation - Syntactic-only
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = SyntacticGAT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2) 
            train(model, dataset, seed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/Syntactic-only GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/Syntactic-only {run} Ablation (seed {seed}).txt", run=run, ablation=True, dataset_class=SyntacticGraphDataset)
            generalization_test(model, dataset, calling_run=run, runs=GENERALIZATION_RUNS, seed=seed, batch_size=BATCH_SIZE, ablation=True, dataset_class=SyntacticGraphDataset)

    #Ablation - Semantic-only
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = SemanticOnlyBERT(num_classes=2, dropout_rate=0.2, freeze_bert_layers=6) # Transfer learning ablation
            train(model, dataset, seed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/Semantic-only GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/Semantic-only {run} Ablation (seed {seed}).txt", run=run, ablation=True, load_if_exist=True, dataset_class=TextDataset)
            generalization_test(model, dataset, calling_run=run, runs=GENERALIZATION_RUNS, seed=seed, batch_size=BATCH_SIZE, ablation=True, dataset_class=TextDataset)
    
    #Ablation - Document GAT
    for run, seed in enumerate(seeds): # Use the same seeds as the normal traning
        set_seed(seed) # Reset the seed every run
        for dataset, vs in [("2WikiMultihopQA", 2500), ("HotpotQA", 1250), ("MuSiQue", 1250)]: # Different validation steps for each dataset since each has different length
            model = DocGATWithBERT(num_classes=2, dropout_rate=0.2, freeze_bert_layers=6) # Transfer learning ablation
            train(model, dataset, seed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_steps=vs, workers=WORKERS, model_save_path=f"../Models/{dataset}/Ablation/Document GAT {run} Ablation (seed {seed}).pt", log_save_path=f"../Logs/{dataset}/Ablation/Document GAT {run} Ablation (seed {seed}).txt", run=run, ablation=True, load_if_exist=True, dataset_class=DocGraphDataset)
            generalization_test(model, dataset, calling_run=run, runs=GENERALIZATION_RUNS, seed=seed, batch_size=BATCH_SIZE, ablation=True, dataset_class=DocGraphDataset)

    # Ablation - Shortcut baselines
    for dataset in tqdm(["2WikiMultihopQA", "HotpotQA", "MuSiQue"]):
        shortcut_validate(dataset, output_path=f"../Logs/{dataset}/Ablation/Shortcut Ablation.txt")