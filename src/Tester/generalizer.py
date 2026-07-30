from ModelTrainer.modelStructures import GraphDataset, convert_to_tuple
import ModelTrainer.trainer as trainer
import Tester.tester as tester
from ModelTrainer.graphFunctions import set_seed

import pandas as pd
import torch
import random

def generalization_test(model, original_dataset: str, calling_run: int, seed: int, runs : int = 5):
    # Define the other datasets
    test_datasets = []

    if original_dataset != "2WikiMultihopQA":
        test_df = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
        test_datasets.append( ("2WikiMultihopQA", GraphDataset(convert_to_tuple(test_df))) )
    if original_dataset != "HotpotQA":
        test_df = pd.read_csv("../Datasets/HotpotQA/test.csv")
        test_datasets.append( ("HotpotQA", GraphDataset(convert_to_tuple(test_df))) )
    if original_dataset != "MuSiQue":
        test_df = pd.read_csv("../Datasets/MuSiQue/test.csv")
        test_datasets.append( ("MuSiQue", GraphDataset(convert_to_tuple(test_df))) )

    # Zero-shot generalization test; just a single run, 
    for test_dataset_name, test_dataset in test_datasets:
        probs, _, acc, cm, _, _, auc_score = tester.test(model, test_dataset)
        pos_probabilities = probs[:, 1]
        y_true = torch.cat([data.pyg_data.y for data in test_dataset], dim=0)
        tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, img_path=f"../Results/{test_dataset_name}/Generalize/[ZERO-SHOT] {original_dataset} generalizing for {test_dataset_name}; {calling_run}.png", title=f"Zero-Shot generalization of {original_dataset} on {test_dataset_name}, {calling_run} (seed {seed}))")

    # Fine-tuned generalization test
    for test_dataset_name, test_dataset in test_datasets:
        train_df = pd.read_csv(f"../Datasets/{test_dataset_name}/train_generalization.csv") # Small training subset just to validate
        train_dataset  = GraphDataset(convert_to_tuple(train_df))

        for run in range(1, runs+1):
            seed = random.randint(1, 101)
            set_seed(seed) # Reset the seed every run
            trainer._train(model, train_dataset, train_dataset, epochs=1, batch_size=32, validation_steps=500, model_save_path=f"../Models/{original_dataset}/Generalize/GAT Generalizing on {test_dataset_name} {run}.pt", log_save_path=f"../Logs/{original_dataset}/Generalize/GAT {run} Ablation (seed {seed}).txt")
            probs, _, acc, cm, _, _, auc_score = tester.test(model, test_dataset)
            pos_probabilities = probs[:, 1]
            y_true = torch.cat([data.pyg_data.y for data in test_dataset], dim=0)
            tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, img_path=f"../Results/{test_dataset_name}/Generalize/[FEW-SHOT] {original_dataset} generalizing for {test_dataset_name}; {run}.png", title=f"Few-Shot generalization of {original_dataset} on {test_dataset_name}, {run} (seed {seed}))")