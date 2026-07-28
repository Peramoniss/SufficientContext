from ModelTrainer.graphFunctions import get_seed

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def _collate(batch: list):
    # Identity collate, the model handles its own batching. Necessary for paralelizing, appearently lambda is not handled well.
    return batch

def test(model, test_dataset, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = TorchDataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,  collate_fn=_collate
    )
    model.eval() # Switch dropout to evaluation behavior

    # Motivational speech to keep me sane
    waiting_sentence = "Testing (rest assured you've done your best. I know, it's hard when you give it your all and the results do not attend your expectations. It feels like effort is pointless because in the end you're going to fail anyway. Remember: failure is inevitable, but it's just a step towards success. You're doing all you can and that's enough ;D)"
    print(waiting_sentence)
    
    complete_probs = []
    complete_correct_labels = []
    
    with torch.no_grad(): # No gradients
        for batch in tqdm(test_loader, desc="Testing the model..", leave=False):
            with torch.amp.autocast(device_type=device.type): # Mixed-precision (FP16) for faster inference
                logits = model(batch)
                probs = torch.softmax(logits, dim=1) # Compute raw probabilities since loss is irrelevant
            
            # Handle probs on CPU since paralelism isn't used anymore
            complete_probs.append(probs.to("cpu"))
            complete_correct_labels.append(torch.cat([item.pyg_data.y for item in batch]).to('cpu'))

        final_probabilities = torch.cat(complete_probs, dim=0).numpy()
        final_true_labels = torch.cat(complete_correct_labels, dim=0).numpy()
        
        
        preds = final_probabilities.argmax(axis=1) # Predictions defined using the max value along the class dimension
        acc = (preds == final_true_labels).sum() / len(final_probabilities)
        cm = confusion_matrix(final_true_labels, preds)
        
        # ROC-AUC for multi-class takes a specific class as the perspective. Since it's binary, taking class 1 (Sufficient) as the reference
        if final_probabilities.shape[1] == 2:
            fpr, tpr, thresholds = roc_curve(final_true_labels, final_probabilities[:, 1])
            area_under_curve = auc(fpr, tpr)
        else:
            fpr, tpr, thresholds, area_under_curve = None, None, None, "Multi-class AUC requires macro/micro strategy using each class as reference"

    return final_probabilities, preds, acc, cm, tpr, thresholds, area_under_curve

def generate_training_dashboard(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    img_path="training_dashboard.png",
    title="Training Dashboard",
    steps_until_val=1,
    automatic_overwrite=False
):
    # Allows the user to not overwrite an important file if they forgot to chose a correct name
    if Path(img_path).is_file() and automatic_overwrite==False:
        choice = input("This filename already exists. How do you want to proceed? [o] Overwrite; [c] Cancel; [r] Rename")
        stop = True
        if choice == 'o':
            stop = False
        elif choice == 'r':
            img_path = input("Insert new name for the file")
            if img_path:
                stop = False
        
        if stop:
            return

    train_losses = np.array(train_losses, dtype=float)
    val_losses   = np.array(val_losses, dtype=float)
    train_accs   = np.array(train_accs, dtype=float)
    val_accs     = np.array(val_accs, dtype=float)

    training_steps = np.arange(1, len(train_losses) + 1) * steps_until_val # Calculate the training steps based on the number of validations and the steps until validation

    # Summary metrics
    best_validation   = int(np.argmin(val_losses)) + 1 # Gets the index of the lowest loss to define the best validation
    best_val_loss = float(np.min(val_losses))
    best_val_acc  = float(val_accs[np.argmin(val_losses)])
    final_train_loss = float(train_losses[-1])
    final_val_loss   = float(val_losses[-1])
    final_train_acc  = float(train_accs[-1]) # TODO: Not used in current version, decide if I'll use it or not
    final_val_acc    = float(val_accs[-1])

    # Normalise accuracy to 0–1 range if supplied as 0–100
    display_acc = lambda v: f"{v:.1%}" if v <= 1.0 else f"{v:.2f}%"

    # Matplotlib layout
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#f7f9fc") # Background color
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.99, color="#1a1a2e") # Dashboard title

    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.35, top=0.85) # Transform the figure in a grid

    # Color scheme
    CARD_BG    = "#f0f4f8"
    CARD_EDGE  = "#c8d6e5"
    TRAIN_PLOT_CLR  = "#2196F3"
    VAL_PLOT_CLR    = "#F4C136"
    GRID_CLR   = "#e0e8f0"

    # Metric cards (top row, all 3 columns)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_cards.axis("off")
    ax_cards.set_facecolor("#f7f9fc")

    # Set the text for the metrics
    end = "th" if best_validation > 3 else ("st" if best_validation == 1 else ("nd" if best_validation == 2 else "rd"))
    metrics = [
        ("Best Validation",      f"{best_validation}{end}"),
        ("Best Val Loss",   f"{best_val_loss:.4f}"),
        ("Best Val Acc",    display_acc(best_val_acc)),
        ("Final Train Loss",f"{final_train_loss:.4f}"),
        ("Final Val Loss",  f"{final_val_loss:.4f}"),
        ("Final Val Acc",  f"{final_val_acc:.4f}"),
        ("Validations Run",      f"{len(training_steps)}"),
    ]

    col_w, card_h = 0.1, 0.12 # Card dimensions
    y = 0.8  # Vertical position from bottom of figure
    x_start = 0.06 # Horizontal position from left to right
    gap = (0.92 - x_start - col_w * len(metrics)) / (len(metrics) - 1) # Calculate gap between cards

    # Print cards
    for i, (label, value) in enumerate(metrics):
        x = x_start + i * (col_w + gap)
        fig.add_artist(plt.Rectangle((x, y), col_w, card_h, # Card rectangle
                    transform=fig.transFigure,
                    facecolor=CARD_BG, edgecolor=CARD_EDGE,
                    linewidth=1.2, clip_on=False))
        fig.text(x + col_w/2, y + card_h*0.65, value, # Value/result text
                transform=fig.transFigure,
                ha='center', va='center', fontsize=15, fontweight='bold', color='#1a1a2e')
        fig.text(x + col_w/2, y + card_h*0.22, label, # Label text
                transform=fig.transFigure,
                ha='center', va='center', fontsize=8.5, color='#666')

    # Loss curves
    ax_loss = fig.add_axes([0.06, 0.08, 0.40, 0.65])
    ax_loss.set_facecolor("#f7f9fc")
    # Train and val curves
    ax_loss.plot(training_steps, train_losses, color=TRAIN_PLOT_CLR, lw=2,   label="Train loss", marker="o", ms=3)
    ax_loss.plot(training_steps, val_losses,   color=VAL_PLOT_CLR,   lw=2,   label="Val loss",   marker="s", ms=3)
    # Dashed vertical line in the best validation 
    ax_loss.axvline(best_validation*steps_until_val, color="#888", linestyle="--", lw=1.2, label=f"Best step ({best_validation*steps_until_val})")
    # Labels
    ax_loss.set_xlabel("Training Steps", fontsize=9)
    ax_loss.set_ylabel("Loss",  fontsize=9)
    ax_loss.set_title("Loss curves", fontsize=12)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, color=GRID_CLR, linewidth=0.7)
    ax_loss.set_xlim([training_steps[0] - 0.5, training_steps[-1] + 0.5])

    # Accuracy curves
    ax_acc  = fig.add_axes([0.55, 0.08, 0.40, 0.65])
    ax_acc.set_facecolor("#f7f9fc")
    ax_acc.plot(training_steps, train_accs, color=TRAIN_PLOT_CLR, lw=2, label="Train acc", marker="o", ms=3)
    ax_acc.plot(training_steps, val_accs,   color=VAL_PLOT_CLR,   lw=2, label="Val acc",   marker="s", ms=3)
    ax_acc.axvline(best_validation*steps_until_val, color="#888", linestyle="--", lw=1.2, label=f"Best step ({best_validation*steps_until_val})")
    ax_acc.set_xlabel("Training Steps",    fontsize=9)
    ax_acc.set_ylabel("Accuracy", fontsize=9)
    ax_acc.set_title("Accuracy curves", fontsize=12)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, color=GRID_CLR, linewidth=0.7)
    ax_acc.set_xlim([training_steps[0] - 0.5, training_steps[-1] + 0.5])

    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()) # Save the image in the defined path
    print(f"Saved → {img_path}")

def generate_test_dashboard(cm, acc, auc_score, probs, y_true, img_path='dashboard.png', title="Model Evaluation Dashboard", automatic_overwrite=False):
    # Allows the user to not overwrite an important file if they forgot to chose a correct name
    if Path(img_path).is_file() and automatic_overwrite == False:
        choice = input("This filename already exists. How do you want to proceed? [o] Overwrite; [c] Cancel; [r] Rename")
        stop = True
        if choice == 'o':
            stop = False
        elif choice == 'r':
            img_path = input("Insert new name for the file")
            if img_path:
                stop = False
        
        if stop:
            return
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1] # Extract data from the confusion matrix

    # Derive metrics from confusion matrix
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Metric cards
    ax_cards = fig.add_subplot(gs[0, :2])
    ax_cards.axis('off')
    metrics = [
        ("Accuracy",  f"{acc:.1%}"),
        ("AUC",       f"{auc_score:.3f}"),
        ("Recall",    f"{recall:.1%}"),
        ("Precision", f"{precision:.1%}"),
        ("F1",        f"{f1:.1%}"),
        ("Samples",   f"{int(tn+fp+fn+tp):,}"),
    ]
    for i, (label, value) in enumerate(metrics):
        x = (i % 3) * 0.34 + 0.01
        y = 0.55 if i < 3 else 0.05
        ax_cards.add_patch(plt.Rectangle((x, y), 0.30, 0.38, # Card rectangle
                            transform=ax_cards.transAxes,
                            facecolor='#f0f4f8', edgecolor='#c8d6e5', linewidth=1,
                            clip_on=False))
        ax_cards.text(x + 0.15, y + 0.26, value, transform=ax_cards.transAxes, # Value text
                    ha='center', va='center', fontsize=15, fontweight='bold', color='#1a1a2e')
        ax_cards.text(x + 0.15, y + 0.10, label, transform=ax_cards.transAxes, # Label text
                    ha='center', va='center', fontsize=9, color='#555')

    # Confusion matrix heatmap (top-right)
    ax_cm = fig.add_subplot(gs[0, 2])
    ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')

    # Configure labels of the confusion matrix
    ax_cm.set_title("Confusion matrix", fontsize=11)
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=9)
    ax_cm.set_yticklabels(['True 0', 'True 1'], fontsize=9)
    labels_cm = [['TN', 'FP'], ['FN', 'TP']]
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{labels_cm[i][j]}\n{cm[i,j]:,}", # Print both the label and the value of each position of the confusion matrix
                    ha='center', va='center', fontsize=10,
                    color='white' if cm[i,j] > thresh else 'black')

    # ROC curve (bottom-left)
    ax_roc = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y_true,  probs)
    ax_roc.plot(fpr,tpr,label=f"auc={auc_score:.2f}")
    # Labels
    ax_roc.set_xlabel('False positive rate', fontsize=9)
    ax_roc.set_ylabel('True positive rate', fontsize=9)
    ax_roc.set_title('ROC curve', fontsize=11)
    ax_roc.legend(fontsize=8)
    ax_roc.set_xlim([-0.02, 1.02]); ax_roc.set_ylim([-0.02, 1.02])

    # Probability histogram (bottom-center)
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(probs.astype(float), bins=20, color='steelblue', edgecolor='white', linewidth=0.5)
    ax_hist.axvline(0.5, color='red', linestyle='--', lw=1.5, label='Threshold 0.5') # Dashed line representing the decision boundary
    ax_hist.set_xlabel('Predicted probability', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.set_title('Probability distribution', fontsize=11)
    ax_hist.legend(fontsize=8)

    # Hits and misses count bar chart (bottom-right)
    ax_bar = fig.add_subplot(gs[1, 2])
    categories = ['TP', 'TN', 'FP', 'FN']
    counts     = [tp, tn, fp, fn]
    colors     = ['#2196F3', '#2196F3', '#F44336', '#F44336']
    bars = ax_bar.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{int(count):,}', ha='center', va='bottom', fontsize=8)
    ax_bar.set_title('Prediction breakdown', fontsize=11)
    ax_bar.set_ylabel('Count', fontsize=9)
    ax_bar.set_ylim(0, max(counts) * 1.12)

    plt.savefig(img_path, dpi=150, bbox_inches='tight') # Save the dashboard in the defined path

    seed = get_seed()
    folder_structure = img_path.rsplit('/', 1)[0] # Split from right to left, stop after one / found. Serves to mark which dataset is being used
    folder_structure = folder_structure + "/results.csv" # Finish the path file
    with open(folder_structure, "a") as f: # Save the results
        f.write(f"{seed}, {acc}, {f1}, {precision}, {recall}, {auc_score}")
        
    print(f"Saved → {img_path}")