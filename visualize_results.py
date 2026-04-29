import json
import matplotlib.pyplot as plt

def plot_learning_curves(history_path):
    # Load the history
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: {history_path} not found. Make sure your training script finished correctly.")
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Custom styling for a premium look
    plt.style.use('seaborn-v0_8-muted') # Standard modern style
    
    plt.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 's-', color='#ff7f0e', linewidth=2, label='Validation Loss')
    
    plt.title('Siamese Network Training Progress (KochNet)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate final values
    if train_loss:
        plt.annotate(f'{train_loss[-1]:.4f}', (len(train_loss), train_loss[-1]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#1f77b4')
    if val_loss:
        plt.annotate(f'{val_loss[-1]:.4f}', (len(val_loss), val_loss[-1]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='#ff7f0e')

    plt.tight_layout()
    
    # Save the figure
    plt.savefig('loss_curve.png', dpi=300)
    print("Graph saved as 'loss_curve.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_learning_curves("training_history.json")
