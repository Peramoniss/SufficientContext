import DatasetGenerator.generate as generator
from ModelTrainer.modelStructures import GraphDataset, SyntacticGraphDataset, TextDataset, DocGraphDataset, convert_to_tuple
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
import gc

# Clean GPU and RAM memory  
def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def _collate(batch: list):
    # Identity collate, the model handles its own batching. Necessary for paralelizing, appearently lambda is not handled well.
    return batch

# Training function, without the abstraction management of train()
def _train(model, train_dataset, val_dataset, epochs=20, lr_bert=2e-5, lr=2e-4, batch_size=8, validation_steps=50, patience=3, workers=0, model_save_path="best_gnn_bert.pt", load_if_exist = False, log_save_path="log.txt"):
    # Setup logging
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
        h.close()

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
        shuffle=True,  collate_fn=_collate,
        num_workers=workers, persistent_workers=True if workers > 0 else False, #True if workers > 0 else False -> took it off since my memory can't support the cache    # keeps workers (and their in-process cache) alive across epochs
        prefetch_factor=2 if workers > 0 else None,          # each worker preloads several batches ahead
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=_collate,
        num_workers=workers//2, persistent_workers=True if workers > 0 else False,    # 
        prefetch_factor=2 if workers > 0 else None,          # each worker preloads several batches ahead
    )
    # print(f"DataLoader batch_size={train_loader.batch_size}, len={len(train_loader)}, dataset_len={len(train_dataset)}")

    # Sets different learning rates for BERT and the GNN to avoid catastrophic forgetting
    embedder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen BERT layers, no point adding them to a group
        if name.startswith("embedder."):
            embedder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": embedder_params, "lr": lr_bert},
        {"params": other_params,    "lr": lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

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
    start_epoch = 0
    start_step_in_epoch = 0

    losses, val_losses = [], []
    accuracies, val_accuracies = [], []
    
    # If the file already exists, resume training (kind of)
    if os.path.exists(model_save_path) and load_if_exist: 
        print("Resuming training...")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        start_epoch = int(start_step // len(train_loader))
        start_step_in_epoch = start_step % len(train_loader)
        epochs_range = range(start_epoch, epochs) # Since train_loader is shuffled, can't return from the same step, only the epoch 
        losses, val_losses = checkpoint['losses'], checkpoint['val_losses']
        accuracies, val_accuracies = checkpoint['accuracies'], checkpoint['val_accuracies']
        best_val_loss = checkpoint['best_val_loss']
        patience_ctr = checkpoint['patience_ctr']

    start_time = time.time()
    for epoch in epochs_range:
        clean_memory() # Memory management
        model.train() # Changes the behavior of dropout to training

        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(42 + epoch)
        train_loader.generator = epoch_generator # Guarantees the shuffling of the dataset will be the same everytime this epoch runs, and that every epoch will have a distinct shuffle

        total_loss, train_correct, train_total = 0.0, 0, 0

        for batch_id, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} train")): # Iterate through batches
            if epoch == start_epoch and batch_id < start_step_in_epoch:
                continue

            optimizer.zero_grad(set_to_none=True) # Restart gradients
            # y = torch.cat([item.pyg_data.y for item in batch]).to(device) # Target values
            y = torch.cat([item.y for item in batch]).to(device) # Target values

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
                        v_y = torch.cat([item.y for item in v_batch]).to(device)
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
                    logger.info(f"  ✓ saved best model at {model_save_path}")
                else: # Otherwise, update the counter for patience (number of validations that the model can go through without improvement)
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info("Early stopping.")
                        return losses, val_losses, accuracies, val_accuracies

                model.train() # Restores training mode
        start_step_in_epoch = 0 # Resets
    training_time = time.time() - start_time
    logger.info(f"Training ended. Duration: {int(training_time // 3600)}h, {int(training_time % 3600 // 60)}min, {int(training_time  % 60)}s")
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
        logger.info(f"  ✓ saved unvalidated model at {model_save_path}")
    del train_loader, val_loader # Guarantees this mess is deleted
    return losses, val_losses, accuracies, val_accuracies

# Abstracted train function
def train(model, dataset: str, seed:int, epochs:int=5, batch_size:int=16, validation_steps:int=2500, patience:int=3, workers:int=0, model_save_path:str='../Models/model.pt', log_save_path:str="../Logs/log.txt", ablation = False, run=1, load_if_exist:bool=False, dataset_class = GraphDataset):
    if not Path("../Datasets/").exists() or not Path(f"../Logs/{dataset}/").exists() or not Path(f"../Models/{dataset}/Generalization/").exists() or not Path(f"../Results/{dataset}/Generalization/").exists(): # If folder structure is incomplete
        # Build it
        datasets = ["2WikiMultihopQA", "HotpotQA", "MuSiQue"]
        for curr_dataset in datasets:
            Path(f"../Datasets/{curr_dataset}").mkdir(parents=True, exist_ok=True)
            Path(f"../Logs/{curr_dataset}/Ablation").mkdir(parents=True, exist_ok=True)
            Path(f"../Models/{curr_dataset}/Ablation").mkdir(parents=True, exist_ok=True)
            Path(f"../Results/{curr_dataset}/Ablation").mkdir(parents=True, exist_ok=True)

        Path(f"../Results/2WikiMultihopQA/Generalize/HotpotQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Results/2WikiMultihopQA/Generalize/MuSiQue").mkdir(parents=True, exist_ok=True)
        Path(f"../Results/HotpotQA/Generalize/2WikiMultihopQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Results/HotpotQA/Generalize/MuSiQue").mkdir(parents=True, exist_ok=True)
        Path(f"../Results/MuSiQue/Generalize/2WikiMultihopQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Results/MuSiQue/Generalize/HotpotQA").mkdir(parents=True, exist_ok=True)

        Path(f"../Models/2WikiMultihopQA/Generalize/HotpotQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Models/2WikiMultihopQA/Generalize/MuSiQue").mkdir(parents=True, exist_ok=True)
        Path(f"../Models/HotpotQA/Generalize/2WikiMultihopQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Models/HotpotQA/Generalize/MuSiQue").mkdir(parents=True, exist_ok=True)
        Path(f"../Models/MuSiQue/Generalize/2WikiMultihopQA").mkdir(parents=True, exist_ok=True)
        Path(f"../Models/MuSiQue/Generalize/HotpotQA").mkdir(parents=True, exist_ok=True)
            

    # Guarantees every dataset was generated (since generalization needs all of them, all of them are needed from the start)
    if not Path("../Datasets/HotpotQA/train.csv").exists(): # If the dataset was not generated yet, generate it
        generator.generate_hotpot_qa_dataset()
    if not Path("../Datasets/2WikiMultihopQA/train.csv").exists():
        generator.generate_2wikimultihop_qa_dataset()
    if not Path("../Datasets/MuSiQue/train.csv").exists():
        generator.generate_musique_dataset()

    # Load datasets depending on the selected one
    if dataset == 'HotpotQA':
        train_df = pd.read_csv("../Datasets/HotpotQA/train.csv")
        val_df = pd.read_csv("../Datasets/HotpotQA/val.csv")
        test_df = pd.read_csv("../Datasets/HotpotQA/test.csv")
    elif dataset == '2WikiMultihopQA':
        train_df = pd.read_csv("../Datasets/2WikiMultihopQA/train.csv")
        val_df = pd.read_csv("../Datasets/2WikiMultihopQA/val.csv")
        test_df = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
    elif dataset == 'MuSiQue':
        train_df = pd.read_csv("../Datasets/MuSiQue/train.csv")
        val_df = pd.read_csv("../Datasets/MuSiQue/val.csv")
        test_df = pd.read_csv("../Datasets/MuSiQue/test.csv")
    elif dataset == 'All':
        train_df = pd.read_csv("../Datasets/2WikiMultihopQA/train.csv")
        val_df = pd.read_csv("../Datasets/2WikiMultihopQA/val.csv")
        test_df = pd.read_csv("../Datasets/2WikiMultihopQA/test.csv")
    
        temp_df = pd.read_csv("../Datasets/HotpotQA/train.csv")
        train_df = pd.concat([train_df, temp_df], ignore_index=True)
        temp_df = pd.read_csv("../Datasets/MuSiQue/train.csv")
        train_df = pd.concat([train_df, temp_df], ignore_index=True)
    
        temp_df = pd.read_csv("../Datasets/HotpotQA/val.csv")
        val_df = pd.concat([val_df, temp_df], ignore_index=True)
        temp_df = pd.read_csv("../Datasets/MuSiQue/val.csv")
        val_df = pd.concat([val_df, temp_df], ignore_index=True)
    
        temp_df = pd.read_csv("../Datasets/HotpotQA/test.csv")
        test_df = pd.concat([test_df, temp_df], ignore_index=True)
        temp_df = pd.read_csv("../Datasets/MuSiQue/test.csv")
        test_df = pd.concat([test_df, temp_df], ignore_index=True)
    else:
        raise ValueError(f'Dataset field is required and must be one of the following: HotpotQA, 2WikiMultihopQA, or MuSiQue. Value sent was {dataset}')
    
    train_dataset = dataset_class(convert_to_tuple(train_df))
    val_dataset   = dataset_class(convert_to_tuple(val_df))
    test_dataset  = dataset_class(convert_to_tuple(test_df))
    
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

    # Train and test the model
    losses, val_losses, accuracies, val_accuracies = _train(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size, validation_steps=validation_steps, patience=patience, workers=workers, model_save_path=model_save_path, log_save_path=log_save_path, load_if_exist=load_if_exist)
    if len(losses) > 0:
        tester.generate_training_dashboard(losses, val_losses, accuracies, val_accuracies, steps_until_val=validation_steps, img_path=f"../Results/{dataset}/{'Ablation/' if ablation else ''}[TRAIN] {dataset} {ablation_type}; {run}.png", title=f"GAT Training ({dataset}, {run})")

    probs, _, y_true, acc, cm, _, _, auc_score = tester.test(model, test_dataset, batch_size=batch_size, workers=workers//2)
    pos_probabilities = probs[:, 1]
    tester.generate_test_dashboard(cm, acc, auc_score, pos_probabilities, y_true, seed=seed, img_path=f"../Results/{dataset}/{'Ablation/' if ablation else ''}[TEST] {ablation_type} {dataset}; {run}.png", title=f"GAT Testing ({dataset}, {run})")