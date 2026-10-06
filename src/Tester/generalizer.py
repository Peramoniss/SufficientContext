from ModelTrainer.modelStructures import GraphDataset, SyntacticGraphDataset, TextDataset, DocGraphDataset, convert_to_tuple
import ModelTrainer.trainer as trainer
import Tester.tester as tester
from ModelTrainer.graphFunctions import set_seed

import pandas as pd
import torch
import random

def generalization_test(model, original_dataset: str, calling_run: int, seed: int, runs : int = 5, batch_size : int = 16, workers : int = 0, ablation=False, dataset_class = GraphDataset):
    # Define the other datasets
    test_datasets = []

    if original_dataset != "2WikiMultihopQA":
        test_df = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
        test_datasets.append( ("2WikiMultihopQA", dataset_class(convert_to_tuple(test_df))) )
    if original_dataset != "HotpotQA":
        test_df = pd.read_csv("../Datasets/HotpotQA/test.csv")
        test_datasets.append( ("HotpotQA", dataset_class(convert_to_tuple(test_df))) )
    if original_dataset != "MuSiQue":
        test_df = pd.read_csv("../Datasets/MuSiQue/test.csv")
        test_datasets.append( ("MuSiQue", dataset_class(convert_to_tuple(test_df))) )

    if ablation:
        if dataset_class == SyntacticGraphDataset:
            ablation_type = "Syntactic"
        elif dataset_class == TextDataset:
            ablation_type = "Semantic"
        elif dataset_class == DocGraphDataset:
                ablation_type = "DocumentGAT"            
        elif dataset_class == GraphDataset and model.freeze_bert_layers >= 12:
            ablation_type = "Transfer"
        else:
            ablation_type = "Undefined Ablation"
    else:
        ablation_type = "SyntacticGAT"

    # Zero-shot generalization test; just a single run, 
    for test_dataset_name, test_dataset in test_datasets:
        probs, _, y_true, acc, cm, _, _, auc_score = tester.test(model, test_dataset, batch_size=batch_size, workers=workers//2)
        pos_probabilities = probs[:, 1]
        tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, seed=seed, img_path=f"../Results/{original_dataset}/Generalize/{test_dataset_name}/[ZERO-SHOT] {ablation_type} on {original_dataset} generalizing; {calling_run}.png", title=f"Zero-Shot generalization of {original_dataset} on {test_dataset_name}, {calling_run} (seed {seed}))")

    # Fine-tuned generalization test
    for test_dataset_name, test_dataset in test_datasets:
        train_df = pd.read_csv(f"../Datasets/{test_dataset_name}/train_generalization.csv") # Small training subset just to validate
        train_dataset  = dataset_class(convert_to_tuple(train_df))

        for run in range(0, runs):
            new_seed = random.randint(1, 101)
            set_seed(new_seed) # Reset the seed every run
            trainer._train(model, train_dataset, train_dataset, epochs=1, batch_size=batch_size, workers=workers, validation_steps=500, model_save_path=f"../Models/{original_dataset}/Generalize/{test_dataset_name}/GAT {ablation_type} Generalizing {run} (pre-train on seed {seed}, fine-tuning on seed {new_seed}).pt", log_save_path=f"../Logs/{original_dataset}/Generalize/GAT {calling_run} - {run} out of {runs} Ablation (seed {seed} - ablation seed {new_seed}).txt")
            probs, _, y_true, acc, cm, _, _, auc_score = tester.test(model, test_dataset, batch_size=batch_size, workers=workers//2)
            pos_probabilities = probs[:, 1]
            tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, seed=seed, img_path=f"../Results/{original_dataset}/Generalize/{test_dataset_name}/[FEW-SHOT] {ablation_type} (seed {seed}, run ({run})); {calling_run}.png", title=f"Few-Shot generalization of {original_dataset} on {test_dataset_name}, {run} (seed {seed}))")