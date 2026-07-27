import DatasetGenerator.generate as generator
import ModelTrainer.graphFunctions as graphFunctions
from ModelTrainer.modelStructures import GraphDataset, convert_to_tuple
import Tester.tester as tester
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path
import time
import logging

def _collate(batch: list):
    # Identity collate, the model handles its own batching. Necessary for paralelizing, appearently lambda is not handled well.
    return batch

# Training function, without the abstraction management of train()
def _train(
    model, train_dataset, val_dataset,
    epochs=20, lr_bert=2e-5, lr=2e-4,
    batch_size=8,
    validation_steps=50,
    patience=3,
    model_save_path="best_gnn_bert.pt", load_if_exist = False,
    log_save_path="log.txt"
):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_save_path, encoding='utf-8'),   # Writes to file
            logging.StreamHandler()           # Writes to console
        ]
    )
    logger = logging.getLogger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = TorchDataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  collate_fn=_collate
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=_collate
    )

    # Sets different learning rates for BERT and the GNN to avoid catastrophic forgetting
    optimizer = torch.optim.AdamW([
        {"params": model.embedder.parameters(), "lr": lr_bert},  # BERT: small
        {"params": model.conv1.parameters(),    "lr": lr},  # GAT: normal
        {"params": model.conv2.parameters(),    "lr": lr},
        {"params": model.mlp.parameters(),      "lr": lr},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler(device=device)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup( # Adapt the learning rate throughout training
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    epochs_range = range(epochs)
    best_val_loss = float("inf")
    patience_ctr  = 0
    steps         = 0

    losses, val_losses = [], []
    accuracies, val_accuracies = [], []
    
    # If the file already exists, resume training (kind of)
    if os.path.exists(model_save_path) and load_if_exist: 
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        start_epoch = int(start_step // len(train_loader))
        epochs_range = range(start_epoch, epochs) # Since train_loader is suffled, can't return from the same step, only the epoch 
        losses, val_losses = checkpoint['losses'], checkpoint['val_losses']
        accuracies, val_accuracies = checkpoint['accuracies'], checkpoint['val_accuracies']
        best_val_loss = checkpoint['best_val_loss']
        patience_ctr = checkpoint['patience_ctr']

    start_time = time.time()
    for epoch in epochs_range:
        graphFunctions.clean_memory() # Memory management
        model.train() # Changes the behavior of dropout to training
        total_loss, train_correct, train_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train"): # Iterate through batches
            optimizer.zero_grad(set_to_none=True) # Restart gradients
            y = torch.cat([item.pyg_data.y for item in batch]).to(device) # Target values

            with torch.amp.autocast(device_type=device.type): # Mixed-precision (FP16) for faster training
                logits = model(batch) # Process the batch
                loss = criterion(logits, y) # Calculate the loss - criterion already applies softmax

            scaler.scale(loss).backward() # Backpropagation considering the scaled loss
            scaler.unscale_(optimizer) # Unscale the gradients before clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Cap the gradients in 1.0 to avoid exploding gradients
            # Update optimizer, scaler, and scheduler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Groups the batch results with the results of other batches
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.shape[0]
            total_loss    += loss.item()
            steps         += 1

            # Periodic validation controlled by validation_steps
            if steps % validation_steps == 0:
                model.eval() # Switch dropout to evaluation behavior
                val_loss, val_correct, val_total = 0.0, 0, 0

                with torch.no_grad(): # No gradients, so no need to build the graph
                    for v_batch in tqdm(val_loader, desc="  val", leave=False):
                        v_y = torch.cat([item.pyg_data.y for item in v_batch]).to(device)
                        with torch.amp.autocast(device_type=device.type):
                            v_logits = model(v_batch)
                            v_loss = criterion(v_logits, v_y)
                        val_loss    += v_loss.item()
                        val_correct += (v_logits.argmax(1) == v_y).sum().item()
                        val_total   += v_y.size(0)

                # Print training results and validation results
                loss = total_loss/validation_steps
                val_loss /= len(val_loader)
                acc = train_correct/train_total
                val_acc = val_correct/val_total
                logger.info(f"\nStep {steps} | Train loss: {loss:.4f} | Val loss: {val_loss:.4f} | Train acc: {acc:.4f} | Val acc: {val_acc:.4f}")

                total_loss, train_correct, train_total = 0.0, 0, 0
                losses.append(loss)
                val_losses.append(val_loss)
                accuracies.append(acc)
                val_accuracies.append(val_acc)

                # If this is the best loss, save the model checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr  = 0 # Restart the patience counter
                    torch.save({
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), # Crucial for resuming!
                        'scheduler_state_dict': scheduler.state_dict(), # Crucial for resuming!
                        'scaler_state_dict': scaler.state_dict(),       # Crucial for mixed precision!
                        'losses': losses,   
                        'val_losses': val_losses,
                        'accuracies': accuracies,
                        'val_accuracies': val_accuracies,
                        'best_val_loss': best_val_loss,
                        'patience_ctr': patience_ctr,
                    }, model_save_path)
                    logger.info("  ✓ saved best model")
                else: # Otherwise, update the counter for patience (number of validations that the model can go through without improvement)
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info("Early stopping.")
                        return losses, val_losses, accuracies, val_accuracies

                model.train() # Restores training mode
    training_time = time.time() - start_time
    logger.info(f"{training_time // 3600}h, {training_time % 3600 // 60}min, {training_time  % 60}s")
    if total_steps < validation_steps: # Didn't validate and didn't save the model
        torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'scaler_state_dict': scaler.state_dict(),
                    'losses': losses,   
                    'val_losses': val_losses,
                    'accuracies': accuracies,
                    'val_accuracies': val_accuracies,
                    'best_val_loss': best_val_loss,
                    'patience_ctr': patience_ctr,
                }, model_save_path)
        logger.info("  ✓ saved model")
    return losses, val_losses, accuracies, val_accuracies

# Abstracted train function
def train(model, dataset: str, epochs:int=5, batch_size:int=16, validation_steps:int=2500, patience:int=3, model_save_path:str='../Models/model.pt', log_save_path:str="../Logs/log.txt", ablation = False, run=1):
    if not Path("../Datasets/").exists() or not Path("../Logs/").exists() or not Path("../Models/").exists() or not Path("../Results/").exists(): # If folder structure is incomplete
        # Build it
        datasets = ["2WikiMultihopQA", "HotpotQA", "MuSiQue"]
        for dataset in datasets:
            Path(f"../Datasets/{dataset}").mkdir(parents=True, exist_ok=True)
            Path(f"../Logs/{dataset}/Ablation").mkdir(parents=True, exist_ok=True)
            Path(f"../Logs/{dataset}/Generalize").mkdir(parents=True, exist_ok=True)
            Path(f"../Models/{dataset}/Ablation").mkdir(parents=True, exist_ok=True)
            Path(f"../Models/{dataset}/Generalize").mkdir(parents=True, exist_ok=True)
            Path(f"../Results/{dataset}/Ablation").mkdir(parents=True, exist_ok=True)
            Path(f"../Results/{dataset}/Generalize").mkdir(parents=True, exist_ok=True)

    # Load datasets depending on the selected one
    if dataset == 'HotpotQA':
        if not Path("../Datasets/HotpotQA/train.csv").exists(): # If the dataset was not generated yet, generate it
            generator.generate_hotpot_qa_dataset()

        train_df = pd.read_csv("../Datasets/HotpotQA/train.csv")
        val_df = pd.read_csv("../Datasets/HotpotQA/val.csv")
        test_df = pd.read_csv("../Datasets/HotpotQA/test.csv")
    elif dataset == '2WikiMultihopQA':
        if not Path("../Datasets/2WikiMultihopQA/train.csv").exists():
            generator.generate_2wikimultihop_qa_dataset()

        train_df = pd.read_csv("../Datasets/2WikiMultihopQA/train.csv")
        val_df = pd.read_csv("../Datasets/2WikiMultihopQA/val.csv")
        test_df = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
    elif dataset == 'MuSiQue':
        if not Path("../Datasets/MuSiQue/train.csv").exists():
            generator.generate_musique_dataset()

        train_df = pd.read_csv("../Datasets/MuSiQue/train.csv")
        val_df = pd.read_csv("../Datasets/MuSiQue/val.csv")
        test_df = pd.read_csv("../Datasets/MuSiQue/test.csv")
    else:
        raise ValueError(f'Dataset field is required and must be one of the following: HotpotQA, 2WikiMultihopQA, or MuSiQue. Value sent was {dataset}')
    
    train_dataset = GraphDataset(convert_to_tuple(train_df))
    val_dataset   = GraphDataset(convert_to_tuple(val_df))
    test_dataset  = GraphDataset(convert_to_tuple(test_df))

    # Train and test the model
    losses, val_losses, accuracies, val_accuracies = _train(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size, validation_steps=validation_steps, patience=patience, model_save_path=model_save_path, log_save_path=log_save_path)
    tester.generate_training_dashboard(losses, val_losses, accuracies, val_accuracies, steps_until_val=2500, img_path=f"../Results/{dataset}/{"Ablation/" if ablation else ''}[TRAIN] {dataset}; {run}.png", title=f"GAT Training ({dataset}, {run})")
    
    probs, _, acc, cm, _, _, auc_score = tester.test(model, test_dataset, batch_size=batch_size*2)
    pos_probabilities = probs[:, 1]
    y_true = torch.cat([data.pyg_data.y for data in test_dataset], dim=0)
    tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, img_path=f"../Results/{dataset}/{"Ablation/" if ablation else ''}[TEST] {dataset}; {run}.png", title=f"GAT Testing ({dataset}, {run})")