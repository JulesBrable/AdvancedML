import matplotlib.pyplot as plt

def plot_optimizer_losses_for_models(models_losses, save=False):
    fig, axes = plt.subplots(1, len(models_losses), figsize=(15, 5))

    for idx, (model_name, optimizers_losses) in enumerate(models_losses.items()):
        ax = axes[idx]
        for opti, losses in optimizers_losses.items():
            ax.plot(losses, label=f'{opti} Optimizer')
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

    plt.suptitle('')
    if save:
        plt.savefig('../assets/img/training_losses_models_optimizers.png')
    plt.tight_layout()
    plt.show()
    