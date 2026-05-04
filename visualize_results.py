import json
import matplotlib.pyplot as plt

def plot_learning_curves(history_path, loss_name="Unknown"):
    # Load the history
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: {history_path} not found. Make sure your training script finished correctly.")
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    epochs = range(1, len(train_loss) + 1)

    # Custom styling for a premium look
    plt.style.use('seaborn-v0_8-muted') 

    # -----------------------------------------
    # Plot 1: Training and Validation Loss
    # -----------------------------------------
    if train_loss and val_loss:
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
        plt.plot(epochs, val_loss, 's-', color='#ff7f0e', linewidth=2, label='Validation Loss')
        
        plt.title(f'Siamese Network Training Progress, Loss: {loss_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Annotate final values
        plt.annotate(f'{train_loss[-1]:.4f}', (len(train_loss), train_loss[-1]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#1f77b4')
        plt.annotate(f'{val_loss[-1]:.4f}', (len(val_loss), val_loss[-1]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#ff7f0e')

        plt.tight_layout()
        loss_filename = f'loss_curve_{loss_name.replace(" ", "_").lower()}.png'
        plt.savefig(loss_filename, dpi=300)
        print(f"Loss graph saved as '{loss_filename}'")
        plt.close()

    # -----------------------------------------
    # Plot 2: Validation Accuracy
    # -----------------------------------------
    if val_acc:
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, val_acc, '^-', color='#2ca02c', linewidth=2, label='Validation Accuracy')
        
        plt.title(f'Siamese Network Validation Accuracy, Loss: {loss_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        
        # Set y-axis slightly wider than the data range for better readability
        min_acc, max_acc = min(val_acc), max(val_acc)
        margin = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 0.1
        plt.ylim(max(0, min_acc - margin), min(1.0, max_acc + margin))
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Annotate final accuracy
        plt.annotate(f'{val_acc[-1]:.4f}', (len(val_acc), val_acc[-1]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#2ca02c')

        plt.tight_layout()
        acc_filename = f'accuracy_curve_{loss_name.replace(" ", "_").lower()}.png'
        plt.savefig(acc_filename, dpi=300)
        print(f"Accuracy graph saved as '{acc_filename}'")
        plt.close()

if __name__ == "__main__":
    plot_learning_curves("history_bce.json", "BCE")
    plot_learning_curves("history_contrastive.json", "Contrastive")
    plot_learning_curves("history_triplet.json", "Triplet")