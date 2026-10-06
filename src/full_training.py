from ModelTrainer.modelStructures import SyntacticGATWithBERT, GraphDataset, SyntacticGAT, SyntacticGraphDataset, SemanticOnlyBERT, TextDataset
from ModelTrainer.trainer import train
from ModelTrainer.graphFunctions import set_seed
from Tester.generalizer import generalization_test
import torch
import pandas as pd

if __name__ == "__main__":
    print("GPU count:", torch.cuda.device_count())
    BATCH_SIZE = 32
    WORKERS = 2
    print("INFO: The tokenizer will be downloaded once per worker. Expect WORKERS+1 calls for hugging face")
    print("INFO: Longer sequence lengths in tokenization are expected and treated within the code")

    seed = 42
    set_seed(seed) # Reset the seed every run

    # model = SyntacticGATWithBERT(hidden_channels=32, num_classes=2, heads=8, dropout_rate=0.2, freeze_bert_layers=6)
    model = SemanticOnlyBERT(num_classes=2, dropout_rate=0.2, freeze_bert_layers=6) 
    print(model._get_number_of_parameters())
    train(model, 'All', seed, epochs=3, batch_size=BATCH_SIZE, validation_steps=5000, workers=WORKERS, model_save_path=f"../Models/All Semantic (seed {seed}).pt", log_save_path=f"../Logs/All Semantic (seed {seed}).txt", run=0, load_if_exist=True, dataset_class=TextDataset)